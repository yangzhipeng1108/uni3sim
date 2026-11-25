"""Observe isolated BF16 linear launch/code facts; never measure GEMM duration."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


SHAPES = ((729, 6144, 2560), (729, 2560, 4096),
    (729, 19456, 2560), (729, 2560, 9728), (1, 151936, 2560))


def observation_shapes(plan_path):
    """Validate an untimed, bounded geometry plan before importing CUDA."""
    if plan_path is None:
        return SHAPES
    plan = json.loads(plan_path.read_text())
    if (plan['no_duration_records'] is not True or plan['no_activity_API'] is not True
            or plan['no_host_API_spans'] is not True
            or type(plan['repeats_per_shape']) is not int or plan['repeats_per_shape'] != 2):
        raise ValueError('module observation requires two untimed repeats per shape')
    shapes = plan['shapes_m_n_k']
    if type(shapes) is not list or not 1 <= len(shapes) <= 32:
        raise ValueError('module observation requires one to32 explicit geometries')
    normalized = []
    for shape in shapes:
        if (type(shape) is not list or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError('module geometry requires three positive integer dimensions')
        m, n, k = shape
        if 2 * (m * k + n * k + m * n) > 2 * 1024**3:
            raise ValueError('module geometry exceeds the two-GiB tensor allocation bound')
        normalized.append(tuple(shape))
    if len(set(normalized)) != len(normalized):
        raise ValueError('module observation contains duplicate geometries')
    return tuple(normalized)


def validate_tensor_map_records(rows, phases):
    """Validate untimed public encode fields without inferring operand roles."""
    fields = {'kind', 'phase', 'api', 'rank', 'data_type', 'global_dimensions',
              'global_strides_bytes', 'box_dimensions', 'element_strides',
              'interleave', 'swizzle', 'l2_promotion', 'oob_fill'}
    result = [row for row in rows if row['kind'] == 'tensor_map']
    for row in result:
        if set(row) != fields or row['api'] != 'cuTensorMapEncodeTiled':
            raise ValueError('tensor-map record requires only the declared untimed fields')
        rank = row['rank']
        if type(rank) is not int or not 1 <= rank <= 5:
            raise ValueError('tensor-map rank must be one to five')
        if type(row['phase']) is not int or not 0 <= row['phase'] < phases:
            raise ValueError('tensor-map phase is outside the bounded capture')
        for field in ('data_type', 'interleave', 'swizzle', 'l2_promotion', 'oob_fill'):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError('tensor-map enum must be a nonnegative integer')
        for field, length in (('global_dimensions', rank), ('global_strides_bytes', rank - 1),
                              ('box_dimensions', rank), ('element_strides', rank)):
            values = row[field]
            if (type(values) is not list or len(values) != length
                    or any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError('tensor-map arrays must match their rank and explicit units')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    shapes = observation_shapes(args.plan)
    expected_cells = 2 * len(shapes)
    args.output.mkdir(exist_ok=False)
    import torch
    torch.cuda.init()
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('this diagnostic requires exactly one authorized H200')
    torch.backends.cuda.matmul.allow_tf32 = False
    library = ctypes.CDLL(str(args.library))
    library.midlayer_gemm_module_init.argtypes = (ctypes.c_char_p, ctypes.c_int)
    library.midlayer_gemm_module_phase.argtypes = (ctypes.c_int,)
    library.midlayer_gemm_module_finish.argtypes = ()
    status = library.midlayer_gemm_module_init(os.fsencode(args.output), 1)
    if status:
        raise RuntimeError(f'failed to initialize module recorder: {status}')
    cells = []
    try:
        for index, (m, n, k) in enumerate(shapes):
            # These exact powers of two yield an exact, full-output BF16 check.
            a = torch.full((m, k), 1 / 32, dtype=torch.bfloat16, device='cuda')
            weight = torch.full((n, k), 1 / 64, dtype=torch.bfloat16, device='cuda')
            torch.cuda.synchronize()
            for repeat in range(2):
                phase = index * 2 + repeat
                if library.midlayer_gemm_module_phase(phase):
                    raise RuntimeError('recorder rejected an earlier callback')
                result = torch.nn.functional.linear(a, weight)
                torch.cuda.synchronize()
                if library.midlayer_gemm_module_phase(-1):
                    raise RuntimeError('module/launch recording failed')
                observed = result.cpu()
                expected = torch.full((m, n), k / 2048, dtype=torch.bfloat16)
                if not torch.equal(observed, expected):
                    raise AssertionError(f'BF16 linear output differs in phase {phase}')
                cells.append({'phase': phase, 'shape_m_n_k': [m, n, k],
                    'input_stride': list(a.stride()), 'weight_stride': list(weight.stride()),
                    'output_shape': list(result.shape), 'output_dtype': str(result.dtype),
                    'expected_element': k / 2048, 'checked_elements': result.numel(),
                    'output_exact': True, 'repeat': repeat})
                del result, observed, expected
            del a, weight
    finally:
        library.midlayer_gemm_module_finish()
        (args.output / 'numerical_cells.json').write_text(json.dumps(cells, indent=2) + '\n')
    metadata = json.loads((args.output / 'module_metadata.json').read_text())
    rows = [json.loads(line) for line in (args.output / 'module_events.jsonl').read_text().splitlines()]
    launches = [row for row in rows if row['kind'] == 'launch']
    tensor_maps = validate_tensor_map_records(rows, expected_cells)
    if (metadata['error_code'] or not metadata['modules'] or len(cells) != expected_cells
            or len(launches) != metadata['launches']
            or set(row['phase'] for row in launches) != set(range(expected_cells))
            or len(tensor_maps) != metadata['tensor_maps']):
        raise ValueError('incomplete bounded module/launch observation')
    libraries = []
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if ('libcublas' in line or 'libcupti' in line) and line.split()[-1].startswith('/')}
    for filename in sorted(paths):
        path = Path(filename)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        libraries.append({'path': filename, 'bytes': path.stat().st_size,
            'sha256': digest})
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name']
        if name is not None and ('cublas' in name.lower() or name.lower() in ('torch', 'vllm')):
            packages.append({'name': name, 'version': distribution.version})
    (args.output / 'receipt.json').write_text(json.dumps({'status': 'complete_candidate',
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'python': platform.python_version(),
        'gpu': torch.cuda.get_device_name(0), 'gpu_cards': 1, 'packages': packages,
        'libraries': libraries, 'cells': len(cells), 'launches': len(launches),
        'modules': metadata['modules'], 'duration_recorded': False,
        'tensor_map_encodes': len(tensor_maps),
        'tensor_map_operand_or_launch_argument_binding_claimed': False,
        'shape_plan_sha256': (hashlib.sha256(args.plan.read_bytes()).hexdigest()
                              if args.plan is not None else None),
        'shapes_m_n_k': shapes,
        'historical_binary_equivalence_proven': False,
        'serving_execution_reproduced': False, 'forward_parameter_adopted': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
