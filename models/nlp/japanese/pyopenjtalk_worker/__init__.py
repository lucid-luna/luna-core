# ====================================================================
#  File: models/nlp/japanese/pyopenjtalk_worker/__init__.py
# ====================================================================
"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

import logging
from typing import Any, Optional

from models.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from models.nlp.japanese.pyopenjtalk_worker.worker_common import WORKER_PORT

logger = logging.getLogger(__name__)

WORKER_CLIENT: Optional[WorkerClient] = None

def run_frontend(text: str) -> list[dict[str, Any]]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
        assert isinstance(ret, list)
        return ret
    else:
        import pyopenjtalk

        return pyopenjtalk.run_frontend(text)

def make_label(njd_features: Any) -> list[str]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
        assert isinstance(ret, list)
        return ret
    else:
        import pyopenjtalk

        return pyopenjtalk.make_label(njd_features)

def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)
    else:
        import pyopenjtalk

        pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)

def update_global_jtalk_with_user_dict(path: str) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)
    else:
        import pyopenjtalk

        pyopenjtalk.update_global_jtalk_with_user_dict(path)

def unset_user_dict() -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")
    else:
        import pyopenjtalk

        pyopenjtalk.unset_user_dict()

def initialize_worker(port: int = WORKER_PORT) -> None:
    import atexit
    import signal
    import socket
    import sys
    import time

    global WORKER_CLIENT
    if WORKER_CLIENT:
        return

    client = None
    try:
        client = WorkerClient(port)
    except (OSError, socket.timeout):
        logger.debug("[L.U.N.A.] Try starting pyopenjtalk worker server")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        count = 0
        while True:
            try:
                client = WorkerClient(port)
                break
            except OSError:
                time.sleep(0.5)
                count += 1
                if count == 20:
                    raise TimeoutError("[L.U.N.A.] 서버가 응답하지 않습니다.")

    logger.debug("[L.U.N.A.] pyopenjtalk worker server started")
    WORKER_CLIENT = client
    atexit.register(terminate_worker)

    def signal_handler(signum: int, frame: Any):
        terminate_worker()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        pass

def terminate_worker() -> None:
    logger.debug("[L.U.N.A.] pyopenjtalk worker server terminated")
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return

    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
