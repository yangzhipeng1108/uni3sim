import os


if "MIDLAYER_STARTUP_MEMORY_RAW" in os.environ:
    try:
        from memory_capture import install

        install()
    except BaseException as error:
        import sys

        sys.stderr.write(
            "Startup memory observer bootstrap failed: " + repr(error) + "\n"
        )
        os._exit(97)
