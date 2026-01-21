import logging
import os.path

# from discord import SyncWebhook  # Import SyncWebhook
import requests

# pip install discord-webhook
WEB_HOOK_URL = r'https://discord.com/api/webhooks/1194669946844762122/334lTQi-65wAO8iqe5MygC_G9M6QVRH2_vNLzrJztDD85cuvZWItuGoK_wBFxACVuOow'
proxies = {
    'http': 'http://127.0.0.1:2080',
    'socket': 'sockets://127.0.0.1:2080'
}


def discord_send(message, file_path=None):
    payload = {
        "content": f"{message}"
    }
    if file_path is not None:
        with open(file_path, 'rb') as f:
            # 文件的内容
            img = f.read()
        file_data = {
            "file": (os.path.basename(file_path), img)
        }
        r = requests.post(WEB_HOOK_URL, data=payload, files=file_data, proxies=proxies, timeout=20)
    else:
        r = requests.post(WEB_HOOK_URL, data=payload, timeout=20, proxies=proxies, )
    return r


class DiscordLogger:
    def __int__(self):
        pass

    def info(self, message, image_path=None):
        discord_send(message, image_path)

    def warm(self, *args, **kwargs):
        return self.info(*args, **kwargs)


if __name__ == '__main__':
    discord_send('ntire2025', file_path=r'E:\projects\business_projects\31098402304.tmp')
#
#
