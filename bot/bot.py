from aiogram import Bot, Dispatcher, Router, F, types, flags, BaseMiddleware
from aiogram.filters import Command, Filter
from  aiogram.enums import ContentType
from aiogram.utils.chat_action import ChatActionMiddleware
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.utils.token import TokenValidationError
from aiogram.exceptions import TelegramUnauthorizedError, TelegramNetworkError
from os import getenv
from sys import exit
import asyncio
from time import sleep
from typing import Union, Callable, Awaitable, Any 
import requests
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types   import ReplyKeyboardRemove, FSInputFile, URLInputFile
import redis


import re

router = Router()
router.message.middleware(ChatActionMiddleware())

BOT_TOKEN = getenv("BOT_TOKEN")
API_URL = getenv('API_URL')
BOT_API_URL = getenv('BOT_API_URL')
REDIS_CREDITS = getenv('REDIS_CREDITS')
if not BOT_TOKEN:
    exit("Error: no bot token provided.\nYou are to set environment variable BOT_TOKEN.\nExititng")
elif not BOT_API_URL:
    exit('Error: no local telegram api server url provided.\n You are to set environment variable BOT_API_URL.\nExiting')
elif not API_URL:
    exit('Error: no API url provided.\n You are to set environment variable API_URL.\nExiting')
elif not REDIS_CREDITS:
    exit('Error: no Redis credits provided.\n You are to set environment variable REDIS_CREDITS.\n export REDIS_CREDITS=<host>:<port>:<password>\nExiting')
try:
    local_api = TelegramAPIServer.from_base(BOT_API_URL)
    session = AiohttpSession(api=local_api)
    bot = Bot(token=BOT_TOKEN, session=session)
    redis_creds = REDIS_CREDITS.split(':')
    r = redis.StrictRedis(
            host = redis_creds[0],
            port = int(redis_creds[1]), 
            password = redis_creds[2]
            )
    storage = RedisStorage(r)

except TokenValidationError:
    exit("Failed to create bot instance: Invalid Bot Token passed!\nExiting...")

 
@router.message(Command(commands= ['start']))
@flags.chat_action('typing')
async def cmd_start(message: types.Message, state:FSMContext):
    await state.set_data({})
    await state.set_state(None)
    await message.answer(f"Hi, {message.from_user.full_name}!\nI'm Video Processing Bot being under (not) active development!\nTry sneding me file")



class PipelineState(StatesGroup):
    #got a file, loaded streams and waiting for next action
    awaiting = State()
    
    choosing_filter = State() 
    
    selecting_streams = State()
    
    choosing_new_stream_type = State()
    choosing_stream_builder = State()
    
    generating_output = State()
    choosing_container = State()

    trim = State()
    stratch = State()
    volume = State()
    lhpass = State()
    rotate = State()
    set_size = State()
    crop = State()
    blur = State()
    scroll = State()
class AttachmentFilter(Filter):
    allowed_content_types = [ContentType.DOCUMENT, 
                             ContentType.VIDEO, 
                             ContentType.AUDIO, 
                             ContentType.VOICE, 
                             ContentType.VIDEO_NOTE,
                             ContentType.ANIMATION
                             ]

    async def __call__(self, message: types.Message, state: FSMContext)->Union[bool, dict]:
        current_state = await state.get_state()
        if message.content_type == ContentType.TEXT and current_state == PipelineState.awaiting:
            return False
        if message.content_type not in self.allowed_content_types:
            await message.reply('This message does not contain any attachment :(')
            return False 
        if message.content_type == ContentType.ANIMATION and message.animation.mime_type == 'image/gif':
            await message.answer('Sorry, but currently i cannot process gif animations :(')
            return False 
        
        attachment = None 
        if message.content_type == ContentType.DOCUMENT:
            attachment = message.document
            if 'video' not in attachment.mime_type:
                await message.answer('This file is not video/audio...')
                return False 
        elif message.content_type == ContentType.VIDEO:
            attachment = message.video
        elif message.content_type == ContentType.AUDIO:
            attachment = message.audio
        elif message.content_type == ContentType.VOICE:
            attachment = message.voice
        elif message.content_type == ContentType.VIDEO_NOTE:
            attachment = message.video_note
        elif message.content_type == ContentType.ANIMATION:
            attachment = message.animation
                
        return {'attachment': attachment}

class StateFilter(Filter):
    def __init__(self, allowed_states: list[State]):
        self.allowed_states = allowed_states
    async def __call__(self, message, state: FSMContext)->bool:
        current_state = await state.get_state()
        return current_state in self.allowed_states





from aiogram.utils import markdown 

async def build_streams_layout(streams_data: dict[str, Any])->str:
    streams_per_file = [f'File [{i}]:\n' for i in range(len(streams_data['files']))]
    new_streams = '\nNEW STREAMS: \n'
    has_new_streams = False
    for i,s in enumerate(streams_data['streams']):
        specs = s['stream_specs']
        filters = [f['filter_name'] for f in s['filters']]
        duration = float(specs['duration'])
        h = int(duration) // (60 * 60)
        m = int(duration) // 60 % 60
        sec = duration - (h*60 - m)*60
        stream_specs = f'\t\t\tduration: {h}:{m}:{sec:.2f}\n\t\t\tbitrate: {specs["bit_rate"]}\n'
        if 'file_ind' in s:
            streams_per_file[ int(s['file_ind'].split(':')[0])  ] += f'\t\t{s["stream_type"].upper()} STREAM [{i}]:  filters: {filters};\n' + stream_specs
        else:
            has_new_streams = True
            source_streams = s['source_streams']
            stream_builder = s['stream_builder']
            new_streams += f'\t\t{s["stream_type"].upper()} STREAM [{i}]: {stream_builder} streams {source_streams}, filters: {filters};\n' + stream_specs
    res = 'FILE STREAMS:\n'
    for s in streams_per_file:
        res += s
    if has_new_streams:
        res += new_streams
    return res
class FileServeMiddleware(BaseMiddleware):
    async def probe(self, file_unique_id: Union[str, int], message: types.Message):
        response = requests.get(API_URL+'/bot/probe_file/'+str(file_unique_id))
        if response.status_code != 200:
            await message.answer('Failed to fetch file info :(\nSkipping that file...')
            return None 
        raw_streams = response.json()['streams']
        return raw_streams

    async def upload_file(self, user_id: Union[int,str], chat_id: Union[int,str], attachment)->bool:
        file_props = {
                'file_id':attachment.file_id,
                'file_unique_id': attachment.file_unique_id,
                'user_id': int(user_id),
                'chat_id': int(chat_id),
                'file_local_path': (await bot.get_file(file_id=attachment.file_id)).file_path
                }
        response = requests.post(API_URL + '/bot/add_file', params=file_props)
        if response.status_code == 200:
            return True
        return False 


    async def __call__(self, handler: Callable[types.Message, Awaitable[Any]], event: types.Message, data: dict[str, Any])->Any:

        if 'attachment' not in data or data['attachment'] is None:
            return await handler(event, data)
        attachment = data['attachment']
        response = requests.get(API_URL + '/bot/check_file/'+str(attachment.file_unique_id))
        file_exists = None
        if response.status_code == 200:
            file_exists = response.json()['file_exists']
        else:
            await event.answer('Some error occured x(\n Please try again later')
            #Cannot provide user with file streams info, so exiting processing
            return
        
        if not file_exists:
            res = await self.upload_file(event.from_user.id, event.chat.id, attachment)
            if not res:
                await event.answer('Failed to fetch file x(\n Please try again later')
                return 

        #now need to probe file 
        raw_streams = await self.probe(attachment.file_unique_id, event)
        if raw_streams is None:
            return
        
        current_streams = await data['state'].get_data()
        #checking if current streams data is empy dict
        if not any(current_streams):
            current_streams = {
                    'files': [],
                    'streams': [],
                    'outputs': []
                    }
        current_streams['files'].append(attachment.file_unique_id)
        for s in raw_streams: 
            s['file_ind'] = f'{len(current_streams["files"])-1}:' +  str(s['file_ind'])
            s['filters'] = []
            current_streams['streams'].append(s)
        await data['state'].set_data(current_streams)
        
        return await handler(event, data)

async def build_awaiting_kb(shape: tuple[int] = (1, 1))->types.ReplyKeyboardMarkup:
    kb_builder = ReplyKeyboardBuilder()
    kb_builder.button(text='Select streams')
    kb_builder.button(text='Create new stream')
    kb_builder.button(text='Generate output')
    kb_builder.button(text='/cancel')
    kb_builder.adjust(*shape)

    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True,
            input_field_placeholder='Select an action...',
            resize_keyboard=True
            )
    return kb_markup


class ActionFilter(Filter):
    def __init__(self, action):
        self.action = action
    async def __call__(self, message: types.Message)->bool:
        if message.text  is not None:
            return message.text.lower() == self.action
        else: 
            return False

router.message.middleware(FileServeMiddleware())


acontainers = ['mp3', 'ac3', 'aac', 'flac', 'm4a', 'ogg']
vcontainers = ['mkv', 'webm' ,'mp4', 'mov', 'flv', 'avi']
available_containers = acontainers + vcontainers

async def build_container_kb():
    
    kb_builder = ReplyKeyboardBuilder()
    for container in available_containers:
        kb_builder.button(text=container)
    kb_builder.adjust(1, 1)
    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True, 
            input_field_placeholder = 'Select container',
            resize_keyboard=True
            )
    return kb_markup

@router.message(StateFilter([PipelineState.choosing_container]), F.text.lower().in_(available_containers))
async def chosen_container_handler(message: types.Message, state: FSMContext):
    container_format = message.text
    data = await state.get_data()
    selected_streams = data.get('selected_streams', [])
    outputs = [{
        'streams': selected_streams, 
        'container_format': container_format,
        'video_bitrate': '512K',
        'audio_bitrate': '128K'
        }]
    data = await state.get_data()
    job = {
        'files': data.get('files', []),
        'streams': data.get('streams', []),
        'outputs': outputs
        }

    res = requests.post(API_URL + '/bot/add_job', json=job)
    if res.status_code != 200:
        await message.answer(text='An error occured with processing your output container :(\n My developer is not a good programmer!', 
                       reply_markup=ReplyKeyboardRemove())
        await state.set_data({})
        await state.set_state(None)
        return
    containers = res.json()
    for container in containers:
        if not container['error']:
            await message.answer_document(caption=container['comment'], document=URLInputFile(API_URL + f'/bot/get_file/{container["file_id"]}', filename=container['filename']), reply_markup=ReplyKeyboardRemove())
        else:
            await message.answer(text=container['comment'], reply_markup=ReplyKeyboardRemove()) 

    await state.set_data({})
    await state.set_state(None)


    #print("POSTED JOB with status ", res.status_code)


@router.message(StateFilter([PipelineState.awaiting]), ActionFilter('generate output'))
async def generate_output_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    streams = data.get('streams', [])
    await state.update_data(generating_output = True)
    await state.set_state(PipelineState.selecting_streams)
    kb = await build_streams_selection_kb(streams = streams)
    await message.answer(text='Select streams which will be added to the output container', reply_markup=kb)

@router.message(Command('cancel'), StateFilter([PipelineState.awaiting]))
async def awaiting_cancel_cmd(message: types.Message, state: FSMContext):
    await state.set_state(None)
    await state.set_data({})
    await message.answer(text='Canceled.\n You still can send me attachments', reply_markup=ReplyKeyboardRemove())


#if message contains animation document is also set
@router.message(StateFilter([None, PipelineState.awaiting]), AttachmentFilter() )
async def get_document_handler(message: types.Message, attachment, state: FSMContext):
    #await message.answer_document(document=attachment.file_id)
    data = await state.get_data()
    caption = await build_streams_layout(data)
    reply_kb = await build_awaiting_kb()
    await message.answer_document(document=attachment.file_id, caption = caption, reply_markup=reply_kb)
    if message.content_type is ContentType.VIDEO_NOTE:
        await message.answer(text=caption, reply_markup=reply_kb)
    await state.set_state(PipelineState.awaiting)

vfilters = ['set size',  'blur', 'edgedetect', 'hflip', 'vflip', 'rotate', 'scroll']
afilters = ['lowpass', 'highpass', 'volume', 'reverb']
avfilters = ['trim', 'stratch', 'reverse']

async def build_filter_kb(streams: list, selected_streams: list = [], shape: tuple[int] = (1, 1))->types.ReplyKeyboardMarkup:
    has_audio = False
    has_video = False
    for i in selected_streams:
        if streams[i]['stream_type'] == 'audio':
            has_audio = True
        if streams[i]['stream_type'] == 'video':
            has_video = True
    sugested_filters = []
    if has_audio and has_video:
        sugested_filters = avfilters
    elif has_audio:
        sugested_filters = afilters + avfilters
    elif has_video:
        sugested_filters = vfilters + avfilters + (['crop'] if len(selected_streams) == 1 else [])


    kb_builder = ReplyKeyboardBuilder()
    for f in sugested_filters:
        kb_builder.button(text=f)
    kb_builder.button(text='done')
    kb_builder.button(text='/cancel')
    kb_builder.adjust(*shape)

    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True,
            input_field_placeholder='Select a filter...',
            resize_keyboard=True
            )
    return kb_markup


async def build_streams_selection_kb(streams: list, selected_streams: list = [], shape: tuple[int] = (1, 1), new_stream_type: str = None)->types.ReplyKeyboardMarkup:
    kb_builder = ReplyKeyboardBuilder()
    for i,s in enumerate(streams):
        if i not in selected_streams:
            if new_stream_type is not None and s['stream_type'] != new_stream_type:
                continue
            kb_builder.button(text = s['stream_type'] + f' stream [{i}]')
    kb_builder.button(text = 'done')
    kb_builder.button(text = '/cancel')
    kb_builder.adjust(*shape)
    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True,
            input_field_placeholder='Select streams...',
            resize_keyboard=True
            )
    return kb_markup

async def build_stream_type_kb()->types.ReplyKeyboardMarkup:
    kb_builder = ReplyKeyboardBuilder()
    kb_builder.button(text='video')
    kb_builder.button(text='audio')
    kb_builder.button(text='/cancel')
    kb_builder.adjust(1, 1)
    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True,
            input_field_placeholder='Choose new stream type..',
            resize_keyboard=True
            )
    return kb_markup




@router.message(Command('cancel'), StateFilter([PipelineState.selecting_streams]))
async def selecting_streams_cancel_cmd(message: types.Message, state: FSMContext):
    data = await state.get_data()
    new_stream_type = data.get('new_stream_type', None)
    await state.update_data(new_stream_type = None, generating_output=False)
    await state.update_data(selected_streams = [])
    kb = None 
    text = None

    if new_stream_type is None:
        await state.set_state(PipelineState.awaiting)
        kb = await build_awaiting_kb() 
        text = await build_streams_layout(await state.get_data())
    else: 
        await state.set_state(PipelineState.choosing_new_stream_type)
        kb = await build_stream_type_kb()
        text = 'Choose new stream type'
    await message.answer(text=text, reply_markup=kb)


@router.message(StateFilter([PipelineState.selecting_streams]), (F.content_type == ContentType.TEXT) & (F.text.lower() != 'done'))
async def stream_selection_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    selected_streams = data.get('selected_streams', [])
    streams = data.get('streams', [])
    tmp  = re.findall(r'\[\d+\]', message.text)
    res = map(lambda x: int(x.replace('[', '').replace(']', '')), tmp)
    if not any(tmp):
        await message.answer(text='Emmm, you think it\' funny to send me crap...')
    new_selected_streams = [i for i in res if i < len(streams) and i not in selected_streams]
    new_stream_type = data.get('new_stream_type', None)
    if new_stream_type is not None:
        new_selected_streams = [i for i in new_selected_streams if streams[i]['stream_type'] == new_stream_type] 
    selected_streams += new_selected_streams
    await state.update_data(selected_streams = selected_streams)
    reply_markup = await build_streams_selection_kb(streams = streams, selected_streams = selected_streams, new_stream_type=new_stream_type)
    await message.answer('You can select more streams', reply_markup=reply_markup)


stream_builders = ['concat', 'mix']

async def build_stream_builder_kb():
    kb_builder = ReplyKeyboardBuilder()
    for i in stream_builders:
        kb_builder.button(text = i)
    kb_builder.button(text = '/cancel')
    kb_builder.adjust(1, 1)
    kb_markup = kb_builder.as_markup(
            one_time_keyboard=True,
            input_field_placeholder='Choose stream builder...',
            resize_keyboard=True
            )
    return kb_markup
#need to process if selected stream for new stream creation
@router.message(StateFilter([PipelineState.selecting_streams]), (F.content_type == ContentType.TEXT) & (F.text.lower() == 'done'))
async def stream_selection_done_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    selected_streams = data.get('selected_streams', [])
    streams = data.get('streams', [])
    if len(selected_streams) == 0:
        await message.answer(text = 'You are to select at least one stream')
        return

    new_stream_type = data.get('new_stream_type', None)
    generating_output = data.get('generating_output', False)
    text = None 
    kb = None 
    state_to_go = None 
    if new_stream_type is None:
        if not generating_output:
            text = await build_streams_layout(await state.get_data())
            text = text + f'\nSelected {len(selected_streams)} streams:\n {selected_streams}\nNow choose filter to apply'
            kb = await build_filter_kb(streams, selected_streams)
            state_to_go = PipelineState.choosing_filter
        else:
            text = 'Select output container type'
            state_to_go = PipelineState.choosing_container
            kb = await build_container_kb()
    else:
        text = 'Choose new stream builder'
        kb = await build_stream_builder_kb() 
        state_to_go = PipelineState.choosing_stream_builder
    await state.set_state(state_to_go)
    await message.answer(text = text, reply_markup = kb)

# in fact just a resolution checking
async def check_streams_compatibility(streams: list[dict], selected_streams: list[int], streams_type: str)->bool:
    if streams_type == 'audio':
        return True
    w = streams[selected_streams[0]]['stream_specs']['width']
    h = streams[selected_streams[0]]['stream_specs']['height']
    for i in selected_streams[1:]:
        if streams[i]['stream_specs']['width'] != w or streams[i]['stream_specs']['height'] != h:
            return False 
    return True


async def concat_streams(streams: list[dict], selected_streams: list[int], stream_type: str)->dict[str, Any]:
    duration = 0.0
    # streams must have same resolution
    w = streams[selected_streams[0]]['stream_specs']['width']
    h = streams[selected_streams[0]]['stream_specs']['height']
    for i in selected_streams: 
        duration += float(streams[i]['stream_specs']['duration'])
    #new stream codec is N/A, ffmpeg will choose it auto
    stream_specs = {'codec': None,
                    'duration': duration,
                    'bit_rate': None,
                    'frame_rate': None, 
                    'sample_rate': None,
                    'width': w, 
                    'height': h
                    }
    new_stream = {'stream_builder':'concat',
                  'source_streams': selected_streams,
                  'stream_type': stream_type,
                  'filters': [],
                  'stream_specs': stream_specs
                  } 
    return new_stream

async def mix_streams(streams: list[dict], selected_streams: list[int], stream_type: str)->dict['str', Any]:
    # streams must have same resolution
    w = streams[selected_streams[0]]['stream_specs']['width']
    h = streams[selected_streams[0]]['stream_specs']['height']
    duration = max([float(streams[i]['stream_specs']['duration']) for i in selected_streams])
    #new stream codec is N/A, ffmpeg will choose it auto
    stream_specs = {'codec': None,
                    'duration': duration,
                    'bit_rate': None,
                    'frame_rate': None, 
                    'sample_rate': None,
                    'width': w, 
                    'height': h
                    }
    new_stream = {'stream_builder':'mix',
                  'source_streams': selected_streams,
                  'stream_type': stream_type,
                  'filters': [],
                  'stream_specs': stream_specs
                  } 
    return new_stream


@router.message(StateFilter([PipelineState.choosing_stream_builder]), F.text.lower().in_(stream_builders))
async def chosen_new_stream_type_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    new_stream_type =  data.get('new_stream_type', None)

    selected_streams_ok = await check_streams_compatibility(streams, selected_streams, new_stream_type)
    if not selected_streams_ok:
        kb = await build_streams_selection_kb(streams, new_stream_type=new_stream_type)
        await message.answer('Selected video streams must have same resolution\n\
                You can change video stream resolution by applying "set size" filter', reply_markup=kb)
        await state.set_state(PipelineState.selecting_streams)
        return 

    new_stream = None
    if message.text.lower() == 'concat':
        new_stream = await concat_streams(streams, selected_streams, new_stream_type)
    elif message.text.lower() == 'mix':
        new_stream = await mix_streams(streams, selected_streams, new_stream_type)
    else: 
        await message.answer('Some error occured')
        return 
    streams.append(new_stream)
    await state.update_data(streams = streams, selected_streams=[], new_stream_type=None)
    await state.set_state(PipelineState.awaiting)
    text = await build_streams_layout(await state.get_data())
    kb = await build_awaiting_kb()
    await message.answer(text=text, reply_markup=kb)

@router.message(StateFilter([PipelineState.awaiting]), ActionFilter('create new stream'))
async def action_create_new_stream_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.choosing_new_stream_type)
    kb = await build_stream_type_kb()
    await message.answer(text='Choose new stream type', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_new_stream_type]), (F.text.lower() == 'audio') | (F.text.lower() == 'video'))
async def choosing_new_stream_type_handler(message: types.Message, state: FSMContext):
    await state.update_data(new_stream_type = message.text.lower())
    await state.set_state(PipelineState.selecting_streams)
    data = await state.get_data()
    streams = data.get('streams', [])
    reply_kb = await build_streams_selection_kb(streams, new_stream_type = message.text.lower())
    await message.answer(text='Now select streams', reply_markup=reply_kb)

@router.message(StateFilter([PipelineState.choosing_new_stream_type]), Command('cancel'))
async def choosing_new_steram_type_cancel_cmd(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.awaiting)
    await state.update_data(new_stream_type = None)
    data = await state.get_data()
    text = await build_streams_layout(data)
    kb = await build_awaiting_kb()
    await message.answer(text=text, reply_markup = kb)

@router.message(StateFilter([PipelineState.awaiting]), ActionFilter('select streams'))
async def action_select_streams_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    await state.update_data(selected_streams = [])
    streams = data.get('streams', [])
    reply_kb = await build_streams_selection_kb(streams)
    await message.answer(text='Now select streams for applying filters',reply_markup=reply_kb)
    await state.set_state(PipelineState.selecting_streams)

@router.message(StateFilter([PipelineState.choosing_filter]), Command('cancel'))
async def choosing_filter_cancel_cmd(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.selecting_streams)
    await state.update_data(selected_filter = None)
    data = await state.get_data()
    kb = await build_streams_selection_kb(streams = data.get('streams', []), selected_streams = data.get('selected_streams', []))
    await message.answer(text = 'Canceled\nYou can select more streams', reply_markup = kb)

@router.message(StateFilter([PipelineState.choosing_filter]), (F.content_type == ContentType.TEXT) & (F.text.lower() == 'done'))
async def choosing_filter_done_handler(message: types.Message, state: FSMContext):
    await state.update_data(selected_streams = [])
    await state.set_state(PipelineState.awaiting)
    text = await build_streams_layout(await state.get_data())
    kb = await build_awaiting_kb()
    await message.answer(text=text, reply_markup=kb)




@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'trim')
async def trim_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.trim)
    await message.answer('''\
How should i trim?\nSend me time interval: start end.
Example: 00:00:00 00:15:00 <-- Keep only first 15 minutes.
stop timestamp is optional.''',
                         reply_markup=types.ReplyKeyboardRemove(), )

def duration_from_str(time: str)->float:
    h, m, s = list(map(float, time.split(':')))[:3]
    return  (h*60 + m)*60 + s

async def add_filter(state, filter_):
    data = await state.get_data()
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    for i in selected_streams:
        streams[i]['filters'].append(filter_)

    await state.update_data(streams = streams)

@router.message(StateFilter([PipelineState.trim]), F.content_type == ContentType.TEXT)
async def trim_params_handler(message: types.Message, state: FSMContext):
    res = re.findall(r'\d+:[0-6]{2}:[0-6]\.?\d+', message.text)[:2]
    start = 0.0
    end = 0.0
    if len(res) == 0:
        message.answer('Wrong time interval...\nTry again')
        return 
    if len(res) < 2:
        start = duration_from_str(res[0])
    else:
        start = duration_from_str(res[0])
        end = duration_from_str(res[1])
    data = await state.get_data() 
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    for i in selected_streams:
        stream_duration = streams[i]['stream_specs']['duration']
        streams[i]['stream_specs']['duration'] = max(end - start, 0.0) if len(res) == 2 else max(float(stream_duration) - start, 0)
        streams[i]['filters'].append({'filter_name':'trim', 'start':start, 'end':end if len(res) == 2 else float(stream_duration)})
    await state.update_data(streams = streams)
    await state.set_state(PipelineState.choosing_filter)
    
    kb = await build_filter_kb(streams, selected_streams)
    await message.answer(text='You can select more filters', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'stratch')
async def stratch_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.stratch)
    await message.answer('''\
Send me a stratch factor:
Values more than 1 will slow stream, while values less than 1 will speed up stream
Example: 0.5
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.stratch]), F.content_type == ContentType.TEXT)
async def stratch_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+\.?\d*', message.text)
    if res is None:
        return 
    stratch_factor = float(res.group())
    data = await state.get_data()
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    for i in selected_streams:
        stream_duration = float(streams[i]['stream_specs']['duration'])
        streams[i]['stream_specs']['duration'] = stream_duration*stratch_factor
        streams[i]['filters'].append({'filter_name':'stratch', 'stratch_factor': stratch_factor})

    await state.update_data(streams = streams)
    await state.set_state(PipelineState.choosing_filter)
    
    kb = await build_filter_kb(streams, selected_streams)
    await message.answer(text='You can select more filters', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'volume')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.volume)
    await message.answer('''\
Send me a volume coefficient:
Values more than 1 will make stream louder, while values less than 1 will decrease stream volume
Example: 0.5
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.volume]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+\.?\d*', message.text)
    if res is None:
        return 
    volume  = float(res.group())

    await add_filter(state, {'filter_name': 'volume', 'volume': volume})
    
    await state.set_state(PipelineState.choosing_filter)
    data = await state.get_data()
    kb = await build_filter_kb(data.get('streams', []), data.get('selected_streams', []))
    await message.answer(text='You can select more filters', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'rotate')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.rotate)
    await message.answer('''\
Send me a rotation angle:
Example: 97.5
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.rotate]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+\.?\d*', message.text)
    if res is None:
        return 
    angle  = float(res.group())

    await add_filter(state, {'filter_name': 'rotate', 'angle': angle})
    
    await state.set_state(PipelineState.choosing_filter)
    data = await state.get_data()
    kb = await build_filter_kb(data.get('streams', []), data.get('selected_streams', []))
    await message.answer(text='You can select more filters', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'blur')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.blur)
    await message.answer('''\
Send me a sigma param for gauss blur:
Example: 1.5
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.blur]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+\.?\d*', message.text)
    if res is None:
        return 
    sigma  = float(res.group())

    await add_filter(state, {'filter_name': 'blur', 'sigma': sigma})
    
    await state.set_state(PipelineState.choosing_filter)
    data = await state.get_data()
    kb = await build_filter_kb(data.get('streams', []), data.get('selected_streams', []))
    await message.answer(text='You can select more filters', reply_markup=kb)

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'set size')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.set_size)
    await message.answer('''\
Send me a resolution: widthxheight
Example:  1920x1080
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.set_size]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+x\d+$', message.text)
    if res is None:
        return 
    w,h  = res.group().split('x')
    if w == 0 or h == 0:
        await message.answer('Zero height or width is not reasonable')
        return 
    data = await state.get_data()
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    for i in selected_streams:
        streams[i]['stream_specs']['width'] = w 
        streams[i]['stream_specs']['height'] = h 
        streams[i]['filters'].append({'filter_name':'set_size', 'height':h, 'width':w})
    
    await state.update_data(streams = streams)
    await state.set_state(PipelineState.choosing_filter)
    kb = await build_filter_kb(streams, selected_streams)
    await message.answer(text='You can select more filters', reply_markup=kb)


@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'crop')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.crop)
    await message.answer('''\
Send me a crop window coordinates (x,y) and window size (widthxheight):
Example:  10 10 100x100
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.crop]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+ \d+ \d+x\d+$', message.text)
    if res is None:
        return 
    x, y, wh  = res.group().split(' ')
    x = int(x)
    y = int(y)
    w, h = wh.split('x')
    w = int(w)
    h = int(h)
    if w == 0 or h == 0:
        await message.answer('Zero height or width is not reasonable')
        return
    data = await state.get_data()
    streams = data.get('streams', [])
    selected_streams = data.get('selected_streams', [])
    
    kb = await build_filter_kb(streams, selected_streams)
    if len(selected_streams) > 1:
        await message.answer('It\'s not possible to apply crop filter to more than one stream', reply_markup=kb)
        await state.set_state(PipelineState.choosing_filter)
        return
    stream = streams[selected_streams[0]]
    if x > int(stream['stream_specs']['width']) or y > int(stream['stream_specs']['height']) or x+w > int(stream['stream_specs']['width']) or y+h > int(stream['stream_specs']['height']):
        await message.answer('Window  is out stream resolution bounds')
        return 

    stream['stream_specs']['width'] = w
    stream['stream_specs']['height'] = h
    stream['filters'].append({'filter_name': 'crop', 'x': x, 'y':y, 'width': w, 'height':h })
    streams[selected_streams[0]] = stream
    await state.update_data(streams = streams)
    await state.set_state(PipelineState.choosing_filter)
    await message.answer(text='You can select more filters', reply_markup=kb)


@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'scroll')
async def volume_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.scroll)
    await message.answer('''\
Send me vertical and horizontal scrolling speed bettween -1 and 1: 
Example: 0 -0.5
WARNING: scrolling speeds are too sensitive, be careful!
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.scroll]), F.content_type == ContentType.TEXT)
async def volume_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^-?\d+\.?\d* -?\d+\.?\d*$', message.text)
    if res is None:
        return 
    h, w  = res.group().split(' ')
    h = float(h)
    w = float(w)

    if not ( -1 <= h <= 1) and not (-1 <= w <= 1):
        await message.answer('Wrong speed parameters')
    
    await add_filter(state, {'filter_name': 'scroll', 'w':w, 'h':h})
    
    await state.set_state(PipelineState.choosing_filter)
    data = await state.get_data()
    kb = await build_filter_kb(data.get('streams', []), data.get('selected_streams', []))
    await message.answer(text='You can select more filters', reply_markup=kb)

    

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower().in_(['lowpass', 'highpass']))
async def lhpass_filter_handler(message: types.Message, state: FSMContext):
    await state.set_state(PipelineState.lhpass)
    await state.update_data(filter_name = message.text)
    await message.answer('''\
Send me a cut frequency:
Example: 600 
''',
                         reply_markup=types.ReplyKeyboardRemove(), )

@router.message(StateFilter([PipelineState.lhpass]), F.content_type == ContentType.TEXT)
async def lhpass_params_handler(message: types.Message, state: FSMContext):
    res = re.search(r'^\d+$', message.text)
    if res is None:
        return 
    mix = 1.0
    freq  = int(res.group())
    data  = await state.get_data()
    await add_filter(state, {'filter_name': data.get('filter_name', 'lowpass'), 'frequency': freq, 'mix':mix})
    
    await state.set_state(PipelineState.choosing_filter)
    data = await state.get_data()
    kb = await build_filter_kb(data.get('streams', []), data.get('selected_streams', []))
    await message.answer(text='You can select more filters', reply_markup=kb)



@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'reverse')
async def reverse_filter_handler(message: types.Message, state: FSMContext):
    await add_filter(state, {'filter_name': 'reverse'})
    await message.answer('Reverse filter applied\nYou can choose more filters from keyboard')

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'reverb')
async def reverb_filter_handler(message: types.Message, state: FSMContext):
    await add_filter(state, {'filter_name': 'reverb'})
    await message.answer('Reverb filter applied\nYou can choose more filters from keyboard')

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'hflip')
async def hflip_filter_handler(message: types.Message, state: FSMContext):
    await add_filter(state, {'filter_name': 'hflip'})
    await message.answer('Hflip filter applied\nYou can choose more filters form keyboard')

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'vflip')
async def vflip_filter_handler(message: types.Message, state: FSMContext):
    await add_filter(state, {'filter_name': 'vflip'})
    await message.answer('Vflip filter applied\nYou can choose more filters form keyboard')

@router.message(StateFilter([PipelineState.choosing_filter]), F.text.lower() == 'edgedetect')
async def edgedetect_filter_handler(message: types.Message, state: FSMContext):
    await add_filter(state, {'filter_name': 'edgedetect'})
    await message.answer('Edgedetect filter applied\nYou can choose more filters form keyboard')





async def main():

    dp  = Dispatcher()
    dp.include_router(router)

    try: 
        await bot.delete_webhook(drop_pending_updates=True)
        print('All updates were skipped!')
        await dp.start_polling(bot)
    except TelegramUnauthorizedError:
        exit('Not valid bot token!')
    except TelegramNetworkError:
        exit('Cannot connect to API server!')




if __name__ == '__main__':
    asyncio.run(main())
