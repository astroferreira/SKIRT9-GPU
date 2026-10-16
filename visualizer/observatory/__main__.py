import argparse

import uvicorn

from .app import create_app

def main():
    parser = argparse.ArgumentParser(description="Run the local SKIRTGPU Observatory")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    allowed_hosts = ["*"] if args.host not in ("127.0.0.1", "::1", "localhost") else None
    uvicorn.run(create_app(allowed_hosts=allowed_hosts), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
