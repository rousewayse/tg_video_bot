from pony import orm
from .pony_cfg import bind_config
db = orm.Database()
db.bind(**bind_config)



