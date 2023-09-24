from .db import db
from pony import orm

class Files(db.Entity):
    _table_ = "files"
    #id should be created automaticly
    # remember that Pony uses empty string instead of None if option nullable is not set to True
    file_id = orm.Required(str, column="file_id", py_check=lambda x : True)
    file_unique_id = orm.Required(str, column="file_unique_id", unique=True)
    user_id = orm.Required(int, column="user_id")
    chat_id = orm.Required(int, column="chat_id")
    #File may not be downloaded
    file_local_path = orm.Optional(str, nullable=True, column="file_local_path")

class LocalFiles(db.Entity):
    _table_ = 'local_files'
    file_id = orm.PrimaryKey(int, auto=True, column='file_id')
    #file_id = orm.Required(int, column='file_id', py_check = lambda x: x>=0, orm.PrimaryKey = True)
    file_local_path = orm.Required(str, nullable=True, column='file_local_path')
    job = orm.Required(lambda: Jobs)
class Jobs(db.Entity):
    _table_ = 'jobs'
    job_id = orm.PrimaryKey(int, auto=True, column='job_id')
    #job_id = orm.Required(int, column="job_id", py_check=lambda x: x>=0)
    body = orm.Required(orm.Json, column='job_body')
    job_done = orm.Required(bool, column='job_done')
    status = orm.Optional(str, column='job_status', nullable = True)
    containers = orm.Set(LocalFiles)
    
