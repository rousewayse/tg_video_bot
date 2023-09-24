from flask import Blueprint, request, Response, jsonify, send_file
import db
from pony import orm
bp  = Blueprint(name="bot",import_name=__name__, url_prefix="/bot/")

@bp.route("/add_file", methods=['POST', 'GET'])
def add_file():
    args = dict(request.args)
    db.add_file(**args)
    return Response(status=200)

@bp.route('/check_file/<string:file_unique_id>', methods=['GET'])
def check_file(file_unique_id: str):
    file_exists = db.check_file(file_unique_id)
    return jsonify({'file_exists': file_exists}) 


from ffmpeg import probe
@bp.route('probe_file/<string:file_unique_id>', methods = ['GET'])
def probe_file(file_unique_id: str):
    file =  db.get_file(file_unique_id) 
    if file is None:
        return Response(status=404)

    parsed_streams = {'streams': []}
    file_streams  = probe(file.file_local_path)['streams']
    for i, s in enumerate(file_streams):
        if s['codec_type'] not in ['video', 'audio']:
            continue
        tmp = {'stream_type': s['codec_type'], 
               'file_ind': i,
               'stream_specs': {
                   'codec': s['codec_name'], 
                   'duration': s['duration'],
                   #i donno why but voice messages have no bit_rate field :(
                   'bit_rate': s['bit_rate'] if 'bit_rate' in s else None, 
                   'frame_rate': s['r_frame_rate'],
                   'height': s['height'] if 'height' in s else None,
                   'width': s['width'] if 'width' in s else None, 
                   'sample_rate': s['sample_rate'] if 'sample_rate' in s else None,
                   }
               } 
        parsed_streams['streams'].append(tmp)

    return jsonify(parsed_streams)

from . import processor
import time
@bp.route('/add_job', methods = ['POST'])
def add_job():
    if request.json is None:
        return Response(status_code=500)
    job_json = request.json
    res = []
    with orm.db_session:
        job = db.entities.Jobs(
                body = job_json,
                job_done = False,
                status = None,
                )
        job_json['files'] = [ db.get_file(file_unique_id=file).file_local_path  for file in job_json['files']]
        processed_job = processor.process(job_json)
        for i in processed_job:
            if not i['error']:
                local_file = db.entities.LocalFiles(file_local_path = i['filepath'], job=job)
                local_file.flush()
                res.append({
                    'file_id':local_file.file_id,
                    'error': i['error'],
                    'comment': i['comment'],
                    'filename': i['filename']
                    })
            else: 
                res.append({
                    'error': i['error'],
                    'comment': i['comment']
                    })
        job.job_done=True
    return jsonify(res), 200

    job = request.json
    files = []
    for file in job['files']:
        files.append(db.get_file(file_unique_id=file).file_local_path)
    job['files'] = files

    res = {'output_file_path': processor.process(job)}
    return jsonify(res), 200


@bp.route('/get_file/<int:local_file_id>', methods=['GET'])
def get_file(local_file_id: int):
    with orm.db_session:
        local_file = db.entities.LocalFiles.get(file_id = local_file_id)
        if local_file is None:
            return Response(404)
        return send_file(local_file.file_local_path, as_attachment=True)
