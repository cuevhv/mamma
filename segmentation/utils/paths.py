import os
import yaml
import json
import platform


def load_yaml_file(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)


def escape_backslashes(path: str) -> str:
    return path.replace("\\", "\\\\")


def create_paths_mapping_unix_to_windows():
    cfg_dict = load_yaml_file(CFG_PATH)
    server_cfg = cfg_dict['server']
    mapping = {
        server_cfg['unix']['ps']: escape_backslashes(server_cfg['windows']['ps']),
        server_cfg['unix']['cluster']: escape_backslashes(server_cfg['windows']['cluster']),
        server_cfg['unix']['fast']: escape_backslashes(server_cfg['windows']['fast']),
    }
    return mapping

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "server_mounts.yaml")
SERVER_MAPPING_U2W = create_paths_mapping_unix_to_windows()
SERVER_MAPPING_W2U = {v: k for k, v in SERVER_MAPPING_U2W.items()}

def string_path_to_windows(path: str):
    if not path:
        return path

    path.replace("\\\\", "\\")
    # Convert to windows path separators
    for unix_path, windows_path in SERVER_MAPPING_U2W.items():
        if path.startswith(unix_path):
            path = path.replace(unix_path, windows_path)
            break

    path = path.replace('/', '\\')
    # In case of double backslashes, replace them with single backslashes
    path = path.replace('\\\\', '\\')
    # Double backslashes only occur in the beginning of the path
    path = path.replace('\\', '\\\\', 1)
    return path


def string_path_to_unix(path):
    if path is None:
        return path
    for windows_path, unix_path in SERVER_MAPPING_W2U.items():
        path = path.replace(windows_path, unix_path)
    # Convert to unix path separators
    path = path.replace('\\', '/').replace("//", "/")
    return path

