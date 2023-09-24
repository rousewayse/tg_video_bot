# Cloud Computing SPbU cource (Telegram Video Processing Bot)

### Configuring
1. Configure database connection in [db/pony_config.py](https://github.com/rousewayse/tg_video_bot/blob/master/db/pony_cfg.py)
2. Configure telgram related tokens in [docker-compose.yaml](https://github.com/rousewayse/tg_video_bot/blob/master/docker-compose.yml)

### Deploying 
```
git clone https://github.com/rousewayse/tg_video_bot.git
cd tg_video_bot
**Configuring**
docker-compose build 
docke-compose up
 ```
### How to use
Start your interaction with bot just by sending it a media:
  - video
  - audio
  - voice message
  - video message
  - document containing audio or video

Then just follow bot's instructions.

Keep in mind that you can send multiple media to bot one after another.

### Bot's menu description
- select streams:
  
    Select media streams (audio or video) for applying ffmpeg filters
- create new stream:

  You can use multiple streams to generate new one (concatinating or mixing streams)

- Generate output

  Done filtering? Then select streams to build output media container. For example, you can cut video's audio by adding only video stream in output container. Then select container format and wait: bot will send you result as far as container would be processed 
