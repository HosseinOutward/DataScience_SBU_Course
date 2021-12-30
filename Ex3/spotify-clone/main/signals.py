from main.models import playlist_song
from django.db.models.signals import pre_save
from django.dispatch import receiver


@receiver(pre_save, sender=playlist_song)
def create_profile(sender, instance, **kwargs):
    # %%
    import requests
    import json

    AUTH_URL = 'https://accounts.spotify.com/api/token'
    # POST
    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': '1452c28790c9401bba6359ea87efe48d',
        'client_secret': '39238108121d4d338a4aeaf9235cffb4',
    })

    # convert the response to JSON
    auth_response_data = auth_response.json()

    # save the access token
    access_token = auth_response_data['access_token']

    # %%
    # Track ID from the URI
    track_id = instance.song_id
    # base URL of all Spotify API endpoints
    BASE_URL = 'https://api.spotify.com/v1/'

    # %%
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    audio_features = r.json()

    # %%
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'tracks/' + track_id, headers=headers)
    r = r.json()
    artist_id = r['artists'][0]['id']

    audio_features['name']=r['name']
    audio_features['image_url']=r['album']['images'][0]['url']

    # %%
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'artists/' + artist_id, headers=headers)
    r = r.json()

    audio_features['genres']=r['genres']
    instance.song_json_data=json.dumps(audio_features)
