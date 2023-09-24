import ffmpeg
from os.path import exists as file_exists



def load_media_container(path: str)->ffmpeg.Stream:
    if not file_exists(path):
        raise FileNotFoundError(f"No such file {path}")
    
    return ffmpeg.input(path)



   
def apply_filter(stream: ffmpeg.nodes.FilterableStream, filter_name: str, *args, **kwargs)-> ffmpeg.nodes.FilterableStream:
    return stream.filter(filter_name, *args, **kwargs) 

def prepare_output(streams: (list, ffmpeg.nodes.FilterableStream), path: str, **output_params)->ffmpeg.nodes.OutputStream:
    streams_ = list()
    if type(streams) == list:
        streams_ = [i.use_stream() if i.use_count != 0 else i.stream  for i in streams]
    elif type(streams) == ffmpeg.nodes.FilterableStream:
        streams_ = [streams.use_stream() if streams.use_count != 0 else streams.stream]
    return ffmpeg.output(*streams_, path, **output_params).overwrite_output()


def run(output: ffmpeg.nodes.OutputStream, quiet: bool = False):
    try:
        output.run(quiet=quiet)
    except ffmpeg._run.Error:
        return False
    return True

def atime_stratch(astream: ffmpeg.nodes.FilterableStream, stretch_factor: float=1.0)->ffmpeg.nodes.FilterableStream:
    if stretch_factor > 2.0:
        return apply_filter(astream, "asetpts", f"{stretch_factor}*PTS")
    else: 
        return apply_filter(astream, 'atempo', f'{1/stretch_factor}')
    #return apply_filter(astream, "asetpts", f"{stretch_factor}*PTS")

def vtime_stratch(vstream: ffmpeg.nodes.FilterableStream, stretch_factor: float=1.0)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "setpts", f'{stretch_factor}*PTS')

def areverse(astream: ffmpeg.nodes.FilterableStream)->ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, 'areverse')

def vreverse(vstream: ffmpeg.nodes.FilterableStream)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, 'reverse')

def atrim(astream: ffmpeg.nodes.FilterableStream, start: str = "00:00:00", end: str = None)->ffmpeg.nodes.FilterableStream:
    tmp = None
    if end is None:
        tmp = apply_filter(astream, "atrim", start=start)
    else: 
        tmp = apply_filter(astream, 'atrim', start=start, end=end)
    return apply_filter(tmp, "asetpts", "PTS-STARTPTS")


def vtrim(vstream: ffmpeg.nodes.FilterableStream, start: str = "00:00:00", end: str = None)->ffmpeg.nodes.FilterableStream:
    tmp = None
    if end is None:
        tmp = apply_filter(vstream, "trim", start=start)
    else:
        tmp = apply_filter(vstream, 'trim', start=start, end=end)
    return apply_filter(tmp, "setpts", "PTS-STARTPTS")

def vset_size(vstream: ffmpeg.nodes.FilterableStream, height: int, width: int)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "scale", height=str(height), width=str(width))

def vset_framerate(vstream: ffmpeg.nodes.FilterableStream, fps: int=24)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "framerate", fps=str(fps))

#def vset_bitrate(vstream: ffmpeg.nodes.FilterableStream, bitrate: int)-> ffmpeg.nodes.FilterableStream:
   
def aresample(astream: ffmpeg.nodes.FilterableStream, rate: int)->ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, "aresample", str(rate))

def ahighpass(astream: ffmpeg.nodes.FilterableStream, frequency: int = 3000, mix: float = 1.0)->ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, "highpass", frequency=frequency, mix=mix)

def alowpass(astream: ffmpeg.nodes.FilterableStream, frequency: int = 500, mix: float = 1.0)->ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, "lowpass", frequency=frequency, mix=mix)

def avolume(astream: ffmpeg.nodes.FilterableStream, volume: float=1.0)->ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, "volume", volume=str(volume))

def areverb(astream: ffmpeg.nodes.FilterableStream, in_gain: float=1.0, out_gain: float=0.7, delays: float=5, decays: float=0.7)-> ffmpeg.nodes.FilterableStream:
    return apply_filter(astream, 'aecho', in_gain=str(in_gain), out_gain=str(out_gain), delays=str(delays), decays=str(decays))

def vcrop(vstream: ffmpeg.nodes.FilterableStream, x: int, y: int, height: int, width: int)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "crop", x=str(x), y=str(y), h=str(height), w=str(width))

def vedgedetect(vstream: ffmpeg.nodes.FilterableStream)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "edgedetect")

def vblur(vstream: ffmpeg.nodes.FilterableStream, sigma:float=0.5)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "gblur", sigma=str(sigma))

def vhflip(vstream: ffmpeg.nodes.FilterableStream)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "hflip")

def vvflip(vstream: ffmpeg.nodes.FilterableStream)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "vflip")

def vrotate(vstream: ffmpeg.nodes.FilterableStream, angle: float)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "rotate", angle=str(angle))

def vscroll(vstream: ffmpeg.nodes.FilterableStream, h: float, v:float)->ffmpeg.nodes.FilterableStream:
    return apply_filter(vstream, "scroll", h=str(h), v=str(v))

def vset_sar(vstream: ffmpeg.nodes.FilterableStream, height: int, width: int)->ffmpeg.nodes.FilterableStream:
    return ffmpeg.filter(vstream, filter_name='setsar', sar=f'{width}/{height}')

def vset_dar(vstream: ffmpeg.nodes.FilterableStream, height: int, width: int)->ffmpeg.nodes.FilterableStream:
    return ffmpeg.filter(vstream, filter_name='setdar', dar=f'{width}/{height}')
#Костыль, чтобы ffmpeg не ныл, что у меня в графе фильтров есть те, которые используются несколько раз... почему он сам это не хэндлит - только ему и известно
class MyStream():
    stream = None
    stream_type = None
    use_count: int = None
    
    codec = None
    duration = None 
    bit_rate = None
    
    #video only props
    width = None
    height = None
    sar = None
    dar = None
    frame_rate = None
    
    #audio only props
    sample_rate = None
    channels = None

    def __init__(self, stream: ffmpeg.nodes.FilterableStream, stream_type = None, **kwargs):
        self.stream = stream
        
        self.stream_type = stream_type
        self.use_count = 0
        for kwarg in kwargs:
            #self.kwarg = kwargs[kwarg]
            setattr(self, kwarg, kwargs[kwarg])
    def use_stream(self)->ffmpeg.nodes.FilterableStream:
        self.use_count += 1
        if self.stream_type == 'video':
            return self.stream.split()[self.use_count]
        elif self.stream_type == 'audio':
            return  self.stream.asplit()[self.use_count]

    def __repr__(self):
        return str( {'stream_type':self.stream_type,'use_count': self.use_count ,'duration':self.duration, 'bit_rate': self.bit_rate }  )
def apply_stream_filters(filters, stream, stream_type):
    res = stream.stream
    for f in filters:
        if f['filter_name'] == 'scroll':
            res = vscroll(res, h=float(f['h']), v=float(f['v']))
        elif f['filter_name'] == 'rotate':
            res = vrotate(res, angle=float(f['angle']))
        elif f['filter_name'] == 'vflip':
            res = vvflip(res) 
        elif f['filter_name'] == 'hflip':
            res = vhflip(res)
        elif f['filter_name'] == 'edgedetect':
            res = vedgedetect(res)
        elif f['filter_name'] == 'blur':
            res = vblur(res, sigma=float(f['sigma']))
        elif f['filter_name'] == 'crop':
            stream.height = int(f['height'])
            stream.width = int(f['width'])
            res = vcrop(res, x=int(f['x']), y=int(f['y']), height=int(f['height']), width=int(f['width']))
        elif f['filter_name'] == 'reverb':
            res = areverb(res)
        elif f['filter_name'] == 'volume':
            res = avolume(res, volume=float(f['volume']))
        elif f['filter_name'] == 'lowpass':
            res = alowpass(res, frequency=int(f['frequency']), mix=float(f['mix']))
        elif f['filter_name'] == 'highpass':
            res = ahighpass(res, frequency=int(f['frequency']), mix=float(f['mix'])) 
        elif f['filter_name'] == 'resample':
            stream.sample_rate = int(f['rate'])
            res = aresample(res, rate=int(f['rate']))
        elif f['filter_name'] == 'set_framerate':
            stream.frame_rate = int(f['fps'])
            res = vset_framerate(res, fps=int(f['fps']))
        elif f['filter_name'] == 'set_size':
            stream.height = int(f['height'])
            stream.width = int(f['width'])
            res = vset_size(res, height=int(f['height']), width=int(f['width']))
        elif f['filter_name'] == 'trim':
            func = vtrim if stream_type == 'video' else atrim 
            if 'end' in f:
                #h, m, s = list(map(float, f['start'].split(':')))[:3]
                #h_, m_, s_ = list(map(float, f['end'].split(':')))[:3]
                #tmp =  ((h_*60 + m_)*60 + s_)  -  ((h*60 + m)*60 + s)
                tmp = f['end'] - f['start']
                stream.duration = tmp
                res = func(res, start=f['start'], end=f['end'])
            else: 
                #h, m, s = list(map(float, f['start'].split(':')))[:3]
                stream.duration -= f['start']
                res = func(res, start=f['start'])
        elif f['filter_name'] == 'reverse':
            func = vreverse if stream_type == 'video' else areverse
            res = func(res)
        elif f['filter_name'] == 'stratch':
            stream.duration *= float(f['stratch_factor'])
            func = vtime_stratch if stream_type == 'video' else atime_stratch
            res  = func(res, stretch_factor=float(f['stratch_factor']))
    stream.stream = res
    return stream


def get_container_metadata(path: str)->dict:
    streams_metadata = ffmpeg.probe(path)['streams']
    res = []
    videos = 0
    audios = 0
    for i,sm in enumerate(streams_metadata):
        if sm['codec_type'] == 'video':
            streams_metadata[i]['stream_ind'] = f'v:{videos}'
            videos += 1
        elif sm['codec_type'] == 'audio':
            streams_metadata[i]['stream_ind'] = f'a:{audios}'
            audios += 1
    return streams_metadata 
    

def build_video_stream_props(metadata):
    res = {}
    res['codec'] = metadata['codec_name']
    res['duration'] = float(metadata.get('duration', 0))
    res['bit_rate'] = int(metadata.get('bit_rate', 0))
    res['width'] = int(metadata.get('width', 0))
    res['height'] = int(metadata.get('height', 0))
    res['sar'] = metadata.get('sample_aspect_ratio', '')
    res['dar'] = metadata.get('display_aspect_ratio', '0')
    rfr  = metadata['r_frame_rate'].split('/')
    rfr = list(map(int, rfr))[:2]
    res['frame_rate'] = float(rfr[0]/rfr[1])
    return res

def build_audio_stream_props(metadata):
    res = {}
    res['codec'] = metadata['codec_name']
    res['duration'] = float(metadata.get('duration', 0))
    res['bit_rate'] = int(metadata.get('bit_rate', 0))
    res['sample_rate'] = int(metadata.get('sample_rate', 0))
    res['channels'] = int(metadata.get('channels', 0))
    return res
 
def get_file_streams(files: list[str], streams, apply_filters = True)->list[MyStream]:
    streams_ = []
    containers = []
    containers_metadata = []
    for s in streams:
        if 'file_ind' not in s: 
            continue
        file_ind, stream_ind = map(int, s['file_ind'].split(':')[0:2])
        if len(containers) <= file_ind:
            containers.append( load_media_container(files[file_ind]))
            containers_metadata.append( get_container_metadata(files[file_ind]))
        metadata_build_func = None
        if s['stream_type'] == 'audio':
            metadata_build_func = build_audio_stream_props
        elif s['stream_type'] == 'video':
            metadata_build_func = build_video_stream_props
        #stream_metadata = metadata_build_func( containers_metadata[file_ind][s['stream_type']][stream_ind]  )
        stream_metadata = metadata_build_func( containers_metadata[file_ind][stream_ind]  )
        
        #stream_ind = s['stream_type'][0] +':' +str(stream_ind)
        stream_ind = containers_metadata[file_ind][stream_ind]['stream_ind']
        streams_.append(MyStream(containers[file_ind][stream_ind], stream_type=s['stream_type'], **stream_metadata))
        if apply_filters:
            streams_[-1] = apply_stream_filters(s['filters'], streams_[-1], s['stream_type'])
    return streams_

def adjust_video_params(vstreams: list[MyStream]): 
    oh = max([i.height for i in vstreams])
    ow = max([i.width for i in vstreams])
    ohsar = max([i.sar.split(':')[1] for i in vstreams])
    owsar = max([i.sar.split(':')[0] for i in vstreams])
    owdar = max([i.dar.split(':')[0] for i in vstreams])
    ohdar = max([i.dar.split(':')[1] for i in vstreams])
    ofr = max([i.frame_rate  for i in vstreams])
    obr = max([i.bit_rate  for i in vstreams])

    #oh = 720
    #ow = 480
    #ohsar=1
    #owsar=1
    #ohdar=9
    #owdar=16
    for i, v in enumerate(vstreams):
        vstreams[i].stream = vset_size(vstreams[i].stream, height=oh, width=ow)
        vstreams[i].stream = vset_sar(vstreams[i].stream, height=ohsar, width=owsar)
        vstreams[i].stream = vset_dar(vstreams[i].stream, height=ohdar, width=owdar)
        vstreams[i].stream = vset_framerate(vstreams[i].stream, fps=ofr)

        vstreams[i].bit_rate = obr
        vstreams[i].frame_rate = ofr
        vstreams[i].height = oh
        vstreams[i].width = ow
        vstreams[i].sar = f'{ohsar}/{owsar}'
        vstreams[i].dar = f'{ohdar}/{owdar}'
    return vstreams



def vmix(vstreams: list[MyStream], duration: str = 'longest')->MyStream:
    
    oduration = max([i.duration for i in vstreams])
    streams = vstreams# adjust_video_params(vstreams)
    ostream = ffmpeg.filter([i.use_stream() for i in streams], filter_name='mix', inputs=len(vstreams), duration=duration)
    ostream_props = {'codec' : None, 
                     'bit_rate': streams[-1].bit_rate, 
                     'width': streams[-1].width, 
                     'height': streams[-1].height, 
                     'sar': streams[-1].sar, 
                     'dar': streams[-1].dar, 
                     'frame_rate': streams[-1].frame_rate,
                     'duration': oduration
                     }
    return MyStream(ostream, stream_type="video", **ostream_props)
    
    

def vconcat(vstreams: list[MyStream])->MyStream:
    oduration = sum( [float(i.duration) for i in vstreams] )
    streams  = vstreams #adjust_video_params(vstreams)
    ostream = ffmpeg.concat(*[i.use_stream() for i in streams], v=1, n=len(streams), a=0, unsafe=True)
    ostream_props = {'codec' : None, 
                     'bit_rate': streams[-1].bit_rate, 
                     'width': streams[-1].width, 
                     'height': streams[-1].height, 
                     'sar': streams[-1].sar, 
                     'dar': streams[-1].dar, 
                     'frame_rate': streams[-1].frame_rate, 
                     'duration': oduration
                     }
    return MyStream(ostream, stream_type='video', **ostream_props)

def adjust_audio_params(astreams: list[MyStream])->list[MyStream]:
    obr = max([i.bit_rate  for i in astreams])
    osr = max([i.sample_rate for i in astreams])
    res = astreams[:]

    for i, v in enumerate(astreams):
        res[i].stream = aresample(res[i].use_stream(), rate=osr)
        res[i].stream = areverse(res[i].stream)
        res[i].bit_rate = obr
        res[i].sample_rate = osr
    return astreams

    

def aconcat(astreams: list[MyStream])->MyStream:
    oduration =  sum( [i.duration for i in astreams] )
    #streams = adjust_audio_params(astreams)
    streams = astreams
    ostream = ffmpeg.concat(*[i.use_stream() for i in streams], v=0, n=len(streams), a=1, unsafe=True)
    ostream_props = {'codec' : None, 
                     'bit_rate': streams[-1].bit_rate, 
                     'sample_rate':streams[-1].sample_rate,
                     'duration': oduration,
                     }
    return MyStream(ostream, stream_type='audio', **ostream_props)

def amix(astreams: list[MyStream], duration: str = 'longest')->MyStream:
    oduration =  max( [i.duration for i in astreams] )
    streams = astreams#adjust_audio_params(astreams)
    ostream = ffmpeg.filter([i.use_stream() for i in streams], filter_name='amix', inputs=len(streams), duration=duration)
    ostream_props = {'codec' : None, 
                     'bit_rate': streams[-1].bit_rate, 
                     'sample_rate':streams[-1].sample_rate,
                     'duration': oduration,
                     }
    return MyStream(ostream, stream_type='audio', **ostream_props)




#not complete
def build_stream(stream: dict, streams)->ffmpeg.nodes.FilterableStream:
    stream_type = stream['stream_type']
    stream_builder = stream['stream_builder']
    res = None
    source_streams = [streams[i] for i in stream['source_streams']]


    if stream_builder == 'mix':
        if stream_type == 'audio':
            res = amix(source_streams, duration='longest')
        elif stream_type == 'video':
            res = vmix(source_streams, duration='longest')
    elif stream_builder == 'concat':
        if stream_type == 'audio':
            res = aconcat(source_streams)
        elif stream_type == 'video':
            res = vconcat(source_streams)
    return apply_stream_filters(stream['filters'], res, stream_type) 
    

def build_virt_streams(file_streams, streams):
    streams_ = file_streams[:]
    for s in streams[len(file_streams):]:
        streams_.append( build_stream(s, streams_) )

    return streams_
        

import shortuuid
import os
def process(job: dict, path_prefix: str = '../target'):
    #filepaths = []
    prefix = os.path.abspath(path_prefix) 
    file_streams = get_file_streams(job['files'], job['streams'])
    virt_streams = build_virt_streams(file_streams, job['streams'])
    res = []
    for o in job['outputs']:
        outfilename = shortuuid.uuid()
        #filepaths.append( prefix+outfilename+'.'+o['container_format'] )
        out_streams = [virt_streams[i] for i in o['streams']]
        tmp_res = {'filepath': prefix+'/'+outfilename+'.'+o['container_format']}
        stderr = None
        try: 
            _, stderr = prepare_output(out_streams, tmp_res['filepath']).run()
            tmp_res['error'] =  False
            tmp_res['comment'] = 'Container processed successfully'
            tmp_res['filename'] = outfilename+'.'+o['container_format']
        except ffmpeg.Error as e:
            tmp_res['error']  = True
            tmp_res['comment'] =  'Processing container failed with ffmpeg error'
        res.append(tmp_res)
    return res 

if __name__ == "__main__":
   pass  
