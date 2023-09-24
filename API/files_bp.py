from flask import Blueprint, request, Response, send_file
import db
from pony import orm
from os.path import exists as file_exists
bp = Blueprint(name="files", import_name=__name__, url_prefix="/files/")


@bp.route("get_file/<string:file_unique_id>", methods=["GET"])
def get_file(file_unique_id: str):
    print("file_unique_id ", file_unique_id)
    with orm.db_session:
        file = db.entities.Files.get(file_unique_id = file_unique_id)
        if file is None:
            return Response(status=404)
        if not file_exists(file.file_local_path):
            return Response(status=404)
        return send_file(file.file_local_path, as_attachment=True)

