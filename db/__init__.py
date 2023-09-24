from pony import orm
from .db import db
from .pony_cfg import generate_mapping_config
from .entities import *
db.generate_mapping(**generate_mapping_config)


@orm.db_session
def get_file(file_unique_id: int):
    return Files.get(file_unique_id = file_unique_id)

@orm.db_session
def add_file(*args, **kwargs):
    file = get_file(kwargs['file_unique_id'])
    if file is not None:
        return file
    return Files(*args, **kwargs)

@orm.db_session
def check_file(file_unique_id: str):
    file = get_file(file_unique_id)
    if file is not None and file.file_local_path is not None:
        return True
    return False










