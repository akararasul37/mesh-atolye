import argparse
import msvcrt
import socket
import threading
import webbrowser

import uvicorn

from .config import RUNTIME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8173, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    host = "127.0.0.1"
    server = uvicorn.Server(uvicorn.Config("mesh_atolye.api:app", host=host, port=args.port))

    if not args.no_browser:
        def open_browser():
            for _ in range(80):
                if server.started:
                    webbrowser.open(f"http://{host}:{args.port}/")
                    return
                threading.Event().wait(0.15)

        threading.Thread(target=open_browser, daemon=True).start()
    with (RUNTIME / "server.lock").open("a+b") as project_lock, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        project_lock.seek(0)
        try:
            msvcrt.locking(project_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise SystemExit("Mesh Atölye bu klasörde zaten çalışıyor.") from exc
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind((host, args.port))
        except OSError as exc:
            raise SystemExit(f"{host}:{args.port} adresi açılamadı. {exc}") from exc
        print(f"Mesh Atölye http://{host}:{args.port}/", flush=True)
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
