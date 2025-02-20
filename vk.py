import binascii
import bs4
import collections
import datetime
import fcntl
import html
import ijson
import itertools
import json
import m3u8
import operator
import os
import pathos
import re
import requests
import scipy.stats
import signal
import subprocess
import time
import traceback
import vk_api.audio
import Crypto.Cipher.AES as aes




def dump_tracks_user(tracks, user_id):
    dump_tracks_file(
        tracks, f'user{user_id}.json'
    )

def dump_tracks_list(tracks, list_id):
    dump_tracks_file(
        tracks, f'list{list_id}.json'
    )

def dump_tracks_file(tracks, file):
    with open(file, 'w') as f:
        json.dump(
            [{
                **({
                    'id': _['id'][1]
                        if isinstance(
                            _['id'], tuple
                            ) else _['id']
                    } if 'id' in _ else {}
                ),
                'artist': _['artist'],
                'title': _['title'],
                **({
                    'subtitle': _['subtitle']
                    } if _.get('subtitle')
                        else {}
                ),
                **({
                    'duration': _['duration']
                    } if 'duration' in _
                        else {}
                )
            } for _ in tracks
                if _['artist'].strip()
                    and _['title'].strip()
            ], f
        )

def track_json_attrs(track, full_id=False):
    return {
        'id': track[0]
            if not full_id else
            (
                str(track[1]),
                str(track[0]),
                track[13].split('/')[2],
                track[13].split('/')[5]
            ),
        'artist': html.unescape(track[4]),
        'title': html.unescape(track[3]),
        'subtitle': html.unescape(track[16]),
        'duration': track[5]
    }




def query_tracks(
    string,
    full_id=False,
    full_list=False
):
    soup = query(string)
    list_href, tracks = \
        parse_block(
            soup,
            'audios',
            full_id
        )
    if full_list and list_href:
        _tracks = \
            obtain_some_tracks(
                list_href, full_id
            )
        if _tracks:
            tracks = _tracks
    if not tracks:
        tracks = \
            list(
                vk_audio.search(
                    string, count=20
                )
            )
    return tracks

def query(string):
    resp = \
        vk_audio._vk.http.post(
            'https://vk.com/al_audio.php',
            data={
                'al': 1,
                'act': 'section',
                'claim': 0,
                'is_layer': 0,
                'owner_id': vk_audio.user_id,
                'section': 'search',
                'q': string
            }
        )
    data = \
        json.loads(
            resp.text.replace('<!--', '')
        )
    return bs4.BeautifulSoup(
        data['payload'][1][0],
        'html.parser'
    )

def parse_block(
    soup,
    kind,
    full_id=False
):
    block = \
        soup.find(
            'div', {
                'class':
                'CatalogBlock__content '
                'CatalogBlock__search'
                    f'_global_{kind}_header '
                'CatalogBlock__layout'
                    '--header'
            }
        )
    if not block:
        return None, []
    link = \
        block.find(
            'a', {
                'class':
                    'audio_page_block'
                        '__show_all_link'
            })
    if link:
        link = \
            'https://vk.com' + link['href']
    tracks = \
        parse_tracks(
            block.parent, full_id
        )
    return link, tracks

def parse_tracks(soup, full_id):
    divs = \
        soup.find_all(
            'div', {'data-audio': True}
        )
    return [
        track_json_attrs(track, full_id)
            for track in map(
                json.loads, map(
                    operator.itemgetter(
                        'data-audio'
                    ), divs
                )
            )
    ]

def obtain_some_tracks(href, full_id=False):
    def handle_signal(number, source):
        raise vk_api.exceptions.AccessDenied()
    signal.signal(
        signal.SIGALRM, handle_signal
    )
    signal.alarm(60)
    try:
        for trial in range(2):
            try:
                resp = vk_audio._vk \
                    .http.post(href)
                soup = bs4.BeautifulSoup(
                    resp.text, 'html.parser'
                )
                return parse_tracks(
                    soup, full_id
                )
            except:
                if not trial:
                    time.sleep(30)
                else:
                    raise
    except vk_api.exceptions.AccessDenied:
        return []
    finally:
        signal.signal(
            signal.SIGALRM, signal.SIG_DFL
        )
        signal.alarm(0)




def obtain_user_tracks(
    user_id,
    tracks_number=None,
    lists_number=100,
    remove_albums=True,
    remove_mixers=True,
    full_id=False
):
    try:
        tracks = []
        if tracks_number and \
            user_id != vk_audio.user_id:
            tracks = \
                obtain_some_tracks(
                    'https://m.vk.com/'
                        f'audios{user_id}'
                )
            if not tracks:
                return []
            if tracks_number is not None \
                and tracks_number <= 100:
                tracks[:] = \
                    tracks[:tracks_number]
            else:
                tracks = []
        if not tracks:
            iter = \
                obtain_user_tracks_fast(
                    user_id,
                    tracks_number,
                    full_id
                )
            tracks = list(iter)
            if not tracks:
                return []
        if tracks_number is not None:
            tracks_number -= len(tracks)
            if tracks_number < \
                0.1 * len(tracks):
                tracks_number = 0
        if (lists_number is None
            or lists_number > 0) and \
            (tracks_number is None
            or tracks_number > 0):
            tracks += \
                obtain_user_list_tracks(
                    user_id,
                    tracks_number,
                    lists_number,
                    remove_albums,
                    remove_mixers,
                    full_id
                )
        return tracks
    except vk_api.exceptions.AccessDenied:
        return []

class Lock:
    def __init__(self, file):
        self.file = file
    def __enter__(self):
        pathlib.Path(self.file).touch()
        self.fp = open(self.file)
        fcntl.flock(
            self.fp.fileno(), fcntl.LOCK_EX
        )
    def __exit__(self, *args):
        fcntl.flock(
            self.fp.fileno(), fcntl.LOCK_UN
        )
        self.fp.close()

def obtain_user_tracks_fast(
    user_id,
    tracks_number=None,
    full_id=False
):
    track_ids = set()
    offset = 0
    while tracks_number is None or \
        offset < tracks_number:
        resp = vk_audio._vk.http.post(
            'https://vk.com/al_audio.php',
            data={
                'al': 1,
                'act': 'load_section',
                'type': 'playlist',
                'owner_id': user_id,
                'playlist_id': -1,
                'offset': offset,
                'is_loading_all': 1
            }
        )
        data = \
            json.loads(
                resp.text.replace('<!--', '')
            )['payload'][1]
        if not data or \
            not data[0] or \
            not data[0]['list']:
            return
        if not tracks_number:
            number = None
        else:
            number = tracks_number - offset
        for track in data[0]['list'][:number]:
            if track[0] in track_ids:
                return
            else:
                track_ids.add(track[0])
            yield track_json_attrs(
                track, full_id
            )
        offset += \
            vk_api.audio.TRACKS_PER_USER_PAGE
        if not data[0]['hasMore']:
            return

def obtain_user_list_tracks(
    user_id,
    tracks_number=None,
    lists_number=None,
    remove_albums=False,
    remove_mixers=False,
    full_id=False
):
    lists = vk_audio.get_albums(user_id)
    lists = [
        _ for _ in lists
            if _['owner_id'] == user_id
                and not auto_list(_)
    ]
    lists.sort(
        key=lambda _list:
            _list.get('plays') or 0
    )
    user_tracks = []
    for _list in lists:
        list_tracks = \
            obtain_list_tracks(
                _list['id'],
                user_id,
                tracks_number,
                full_id
            )
        if remove_albums:
            tracks = \
                load_tracks_json(
                    list_tracks
                )
            artist = {
                track_artist(track)[0]
                    for track in tracks
            }
            if len(artist) == 1 and \
                len(tracks) > 1:
                continue
        if remove_mixers:
            if len(list_tracks) == 100:
                continue
        user_tracks.append(list_tracks)
        if tracks_number is not None:
            if len(list_tracks) < \
                tracks_number:
                tracks_number -= \
                    len(list_tracks)
            else:
                break
        if not remove_albums and \
            not remove_mixers:
            if lists_number:
                lists_number -= 1
                if not lists_number:
                    break
    if lists_number:
        user_tracks.sort(
            key=len, reverse=True
        )
        user_tracks[:] = \
            user_tracks[:lists_number]
    return list(
        itertools.chain(*user_tracks)
    )

def obtain_list_tracks(
    list_id,
    user_id=None,
    tracks_number=None,
    full_id=False
):
    if isinstance(list_id, str):
        user_id, list_id = \
            user_list(list_id)
    url = \
        'https://vk.com' \
            '/music/playlist' \
                f'/{user_id}_{list_id}'
    resp = vk_audio._vk.http.post(url)
    if not resp.ok:
        print(
            resp.url,
            resp.status_code,
            resp.reason
        )
        return []
    if resp.url != url:
        return []
    start_token, final_token = \
        '"list":[[', ']]}'
    other, *items = \
        resp.text.split(start_token, 1)
    if not items:
        return []
    items, *other = \
        items.pop().split(final_token, 1)
    assert other
    tracks = \
        json.loads('[[' + items + ']]')
    return [
        track_json_attrs(track, full_id)
            for track in tracks
        ][:tracks_number]




def user_list(id):
    re_list_id = \
        re.compile(r'(-?\d+)_(\d+)?(_\w+)?')
    result = re_list_id.search(id)
    if result:
        owner_id, album_id, _ = \
            result.groups()
        return int(owner_id), int(album_id)

def user_list_id(id):
    list_href_path = (
        '/music/playlist/', '/music/album/'
    )
    if id.startswith(list_href_path):
        owner_id, album_id = \
            user_list(
                id[id.rindex('/') + 1:]
            )
        return f'{owner_id}_{album_id}'

def auto_list(_):
    if _['title'] in (
        'Для вас',
        'Плейлист дня',
        'Плейлист недели'
    ):
        return True
    if re.search(
        r'\d{2}.\d{2}.\d{4}',
        _['title']
    ):
        return True
    return False

def query_lists(string):
    soup = query(string)
    list_page_href, _ = \
        parse_block(soup, 'playlists')
    slider = \
        soup.find(
            'div', {
                'class':
                    'CatalogBlock__content '
                    'CatalogBlock__search'
                        '_global_playlists '
                    'CatalogBlock__layout'
                        '--large_slider'
            }
        )
    link = \
        slider.find_all(
            'a', {'class': 'audio_pl__cover'}
            ) if slider else []
    list_uids = \
        list(
            map(
                user_list_id,
                map(
                    operator \
                        .itemgetter('href'),
                    link
                )
            )
        )
    return list_page_href, list_uids

def fetch_lists(list_page_href):
    resp = vk_audio._vk.http.post(
        list_page_href
    )
    def attr(text, name):
        return text \
            .split(f'"{name}":"')[1] \
                .split('"', 1)[0]
    section_id = attr(resp.text, 'sectionId')
    next_from = None
    while True:
        resp = vk_audio._vk.http.post(
            'https://vk.com/al_audio.php',
            data={
                'al': 1,
                'act': 'load_catalog_section',
                'section_id': section_id,
                **({
                    'start_from': next_from
                    } if next_from else {}
                )
            }
        )
        data = json.loads(
            resp.text.replace('<!--', '')
        )
        soup = bs4.BeautifulSoup(
            data['payload'][1][0][0],
            'html.parser'
        )
        link = soup.find_all(
            'a', {'class': 'audio_pl__cover'}
        )
        yield from map(
            user_list_id,
            map(
                operator.itemgetter('href'),
                link
            )
        )
        section_id = \
            data['payload'][1][1]['sectionId']
        next_from = \
            data['payload'][1][1]['nextFrom']
        if not next_from:
            break

def return_track_lists(
    tracks,
    obtain=False,
    update=False,
    slicer=slice(None),
    factor=0.5
):
    list_file = f'user{client}list.json'
    if os.path.isfile(list_file):
        with open(list_file, 'r') as f:
            track_lists = json.load(f)
    else:
        track_lists = {}
    if not obtain and not update:
        return track_lists
    for index, track in \
        list(enumerate(tracks))[slicer]:
        if track not in track_lists:
            if not obtain:
                continue
        else:
            if not update:
                continue
        aliases = [track]
        artist = track_artist(track)
        if artist[1:]:
            aliases.append(
                track_id(
                    artist=artist[0],
                    title=track_title(track)
                )
            )
        lists = \
            set(
                track_lists.get(track, [])
            )
        _lists = lists.copy()
        for alias in aliases:
            if alias != track:
                alias = html.unescape(alias)
                alias = fixed_string(alias)
                if alias == track:
                    continue
            list_uids = set()
            list_uids_iter = None
            list_page_href, next_list_uids = \
                query_lists(alias)
            next_list_uids = set(next_list_uids)
            while True:
                _factor = (
                    len(
                        next_list_uids - lists
                        ) / len(next_list_uids)
                    ) if next_list_uids else 0
                if _factor < factor:
                    break
                list_uids.update(
                    next_list_uids
                )
                if not list_uids_iter:
                    lists_batch_limit = 6
                    if list_page_href and \
                        len(next_list_uids) >= \
                            lists_batch_limit:
                        list_uids_iter = \
                            fetch_lists(
                                list_page_href
                            )
                    else:
                        break
                lists_batch = 20
                next_list_uids = \
                    set(
                        itertools.islice(
                            list_uids_iter,
                            lists_batch
                        )
                    )
            print(
                f"'{track}': '{alias}'",
                len(list_uids),
                len(lists),
                len(lists | list_uids)
            )
            lists.update(list_uids)
        print(
            index,
            f"'{track}'",
            len(_lists),
            len(_lists | lists)
        )
        track_lists[track] = list(lists)
        with open(list_file, 'w') as f:
            json.dump(track_lists, f)
    return track_lists




def obtain_list_source(
    track_lists,
    source_number=None,
    source_outset=None
):
    tracks_list_counts = \
        collections.Counter(
            itertools.chain(
                *track_lists.values()
            )
        )
    obtain_source_tracks(
        source_weight=tracks_list_counts,
        source_number=source_number,
        source_outset=source_outset
    )

def obtain_list_user_source(
    track_lists,
    source_number=None,
    source_outset=None
):
    user_tracks_list = {}
    for list_id in tracks_list_counts:
        file = os.path.join(
            'groups',
            f'user{client}list',
            f'list{list_id}.json'
        )
        if file in common_tracks:
            user_id, list_id = \
                user_list(list_id)
            user_tracks_list \
                .setdefault(
                    user_id, set()
                    ).update(
                        common_tracks[file]
                    )
    tracks_list_user_counts = \
        collections.Counter({
            user_id: len(tracks)
                for user_id, tracks in
                    user_tracks_list.items()
        })
    remove_groups(
        tracks_list_user_counts
    )
    obtain_source_tracks(
        source_weight=
            tracks_list_user_counts,
        source_number=source_number,
        source_outset=source_outset
    )

def obtain_user_source(
    track_lists,
    source_number=None,
    source_outset=None
):
    tracks_user_counts = \
        collections.Counter(
            itertools.chain.from_iterable(
                set(
                    map(
                        lambda list_id:
                            user_list
                                (list_id)[0],
                        lists
                    )
                ) for lists in
                    track_lists.values()
            )
        )
    remove_groups(tracks_user_counts)
    obtain_source_tracks(
        source_weight=tracks_user_counts,
        source_number=source_number,
        source_outset=source_outset
    )

def remove_groups(counts):
    def is_group_id(user_id):
        return user_id < 0
    for user_id in list(counts):
        if is_group_id(user_id):
            del counts[user_id]




def obtain_group_users(
    group_id,
    update=False,
    fields=None
):
    path = 'users'
    os.makedirs(path, exist_ok=True)
    file = os.path.join(
        path, f'users-{group_id}.json'
    )
    if not update \
        and os.path.exists(file):
        with open(file, 'r') as f:
            users = json.load(f)
            if not fields:
                users = [
                    _['id']
                        if type(_) == dict
                            else _
                        for _ in users
                ]
            return users
    users_max_batch = 1000
    user_ids = \
        vk_tools.get_all(
            'groups.getMembers',
            users_max_batch,
            {'group_id': group_id}
        )['items']
    if not fields:
        with open(file, 'w') as f:
            json.dump(user_ids, f)
        return user_ids
    users = []
    while True:
        users_batch = \
            vk.users.get(
                user_ids=','.join(
                    map(
                        str,
                        user_ids[
                            len(users):
                            len(users) +
                            users_max_batch
                        ]
                    )
                ),
                fields=','.join(fields)
            )
        users.extend(
            map(
                lambda user: {
                    attr: user[attr]
                        for attr in fields
                        if user.get(attr, '')
                }, users_batch
            )
        )
        if len(users_batch) < \
            users_max_batch:
            break
    with open(file, 'w') as f:
        json.dump(users, f)
    return users

def obtain_group_posts(
    group_id,
    update=False
):
    path = 'posts'
    os.makedirs(path, exist_ok=True)
    file = \
        os.path.join(
            path, f'posts-{group_id}.json'
        )
    if os.path.exists(file):
        with open(file, 'r') as f:
            posts = json.load(f)
    else:
        posts = []
    if update:
        posts_added = []
        posts_batch = 25
        try:
            for post in vk_tools \
                .get_all_iter(
                    'wall.get',
                    posts_batch,
                    {'owner_id': -group_id}
                ):
                if post.get('is_pinned'):
                    continue
                if posts and \
                    post['date'] <= \
                        posts[-1]['date']:
                        break
                print(
                    len(posts_added),
                    post['id']
                )
                posts_added.append(post)
        except Exception as e:
            traceback.print_exc()
        posts.extend(posts_added[::-1])
        with open(file, 'w') as f:
            json.dump(posts, f)
    return posts

def obtain_group_likes(
    group_id,
    update=False
):
    path = 'likes'
    os.makedirs(path, exist_ok=True)
    file = \
        os.path.join(
            path, f'likes-{group_id}.json'
        )
    if os.path.exists(file):
        with open(file, 'r') as f:
            likes = {
                int(post_id)
                    if isinstance(
                        post_id, str
                    ) else post_id
                    : likers
                    for post_id, likers in
                        json.load(f).items()
            }
        if not update:
            return likes
    else:
        likes = {}
    posts = obtain_group_posts(group_id)
    while len(likes) < len(posts):
        posts_max_batch = 1000
        likes_max_batch = 1000
        likes_batch, errors = \
            vk_api.vk_request_one_param_pool(
                vk_session,
                'likes.getList',
                key='item_id',
                values=list(map(
                    operator.itemgetter('id'),
                    posts[
                        len(likes):
                        len(likes) +
                            posts_max_batch
                    ]
                )),
                default_values={
                    'type': 'post',
                    'owner_id': -group_id,
                    'count': likes_max_batch
                }
            )
        assert not errors, errors
        likes_added = {
            int(post_id): likers['items']
                for post_id, likers
                    in likes_batch.items()
                if int(post_id) not in likes
        }
        likes.update(likes_added)
        with open(file, 'w') as f:
            json.dump(likes, f)
        if not likes_added:
            break
    return likes




def obtain_active_source(
    group_id,
    source_prefix=None,
    tracks_filter=None,
    tracks_subset=None,
    fscore_weight=None,
    **kwargs
):
    posts = \
        load_source_page(
            -group_id, flat=False
        )
    if not posts:
        obtain_group_posts(
            group_id, update=True
        )
        posts = \
            load_source_page(
                -group_id, flat=False
            )
    likes = \
        obtain_group_likes(group_id)
    if not likes:
        likes = \
            obtain_group_likes(
                group_id, update=True
            )
    if tracks_filter is True:
        tracks_counts = \
            collections.Counter(
                itertools.chain(
                    *posts.values()
                )
            )
        tracks_filter = \
            dict(
                tracks_counts
                    .most_common(
                        tracks_subset
                    )
            )
    source_counts = \
        collections.Counter(
            itertools.chain(
                *likes.values()
            )
        )
    if not tracks_filter:
        source_weight = source_counts
    else:
        if not isinstance(
            tracks_filter, (set, dict)
        ):
            tracks_filter = set(tracks_filter)
        source_scores = {}
        for post_id, tracks in posts.items():
            scores = collections.Counter({
                track: 1 / len(tracks)
                    for track in tracks
                        if not tracks_filter or
                            track in tracks_filter
            })
            if scores:
                for source in likes[post_id]:
                    source_scores \
                        .setdefault(
                            source,
                            collections.Counter()
                         ).update(scores)
        def weight(source):
            common = \
                sum(
                    min(score, 1)
                        for score in
                            source_scores
                                [source].values()
                )
            if not fscore_weight:
                return (
                    numpy.round(common, 6),
                    source_counts[source]
                )
            precis = \
                common / source_counts[source]
            recall = \
                common / len(tracks_filter)
            fscore = \
                scipy.stats.hmean(
                    (precis, recall),
                    weights=(
                        1 / fscore_weight**2,
                        1
                    )
                )
            return (
                numpy.round(fscore, 6),
                numpy.round(common, 6),
                source_counts[source]
            )
        source_weight = \
            collections.Counter({
                source: weight(source)
                    for source in source_scores
            })
    if client in source_weight:
        del source_weight[client]
    obtain_source_tracks(
        source_prefix=source_prefix or
            f'groups/{group_id}',
        source_weight=source_weight,
        **kwargs
    )

def obtain_probed_source(
    source_prefix=None,
    source_number=None,
    probes_prefix=None,
    probes_number=None,
    source_metric=None,
    tracks_metric=None,
    fscore_weight=1,
    public_tracks=None,
    client_tracks=None,
    metric_merger=
        scipy.stats.hmean,
    **kwargs
):
    if probes_number != 0:
        obtain_source_tracks(
            source_prefix=probes_prefix,
            source_number=probes_number,
            tracks_number=100
        )
    if public_tracks and \
        not isinstance(
            public_tracks, set
        ):
        public_tracks = set(public_tracks)
    if client_tracks and \
        not isinstance(
            client_tracks, set
        ):
        client_tracks = set(client_tracks)
    if not source_metric:
        def source_metric(tracks):
            tracks = set(tracks)
            if public_tracks:
                public_metric = \
                    tracks_metric(
                        tracks, public_tracks
                    )
            if client_tracks:
                client_metric = \
                    tracks_metric(
                        tracks, client_tracks
                    )
            if public_tracks \
                and client_tracks:
                return metric_merger((
                    public_metric,
                    client_metric
                ))
            elif public_tracks:
                return public_metric
            elif client_tracks:
                return client_metric
            else:
                assert f'{public_tracks=} ' \
                    f'{client_tracks=}'
    if not tracks_metric:
        def tracks_metric(tracks1, tracks2):
            common = \
                len(
                    set(
                        map(
                            lambda _:
                            track_artist(_)[0],
                            tracks1 & tracks2
                        )
                    )
                )
            if fscore_weight:
                if common == 0:
                    return 0
                precis = common / len(tracks1)
                recall = common / len(tracks2)
                return scipy.stats.hmean(
                    (precis, recall),
                    weights=(
                        1 / fscore_weight**2, 1
                    )
                )
            else:
                return common
    def source_weight():
        weight = collections.Counter()
        def tracks_reduce(result):
            for source, tracks in result:
                id = source_id(source)
                weight[id] = \
                    source_metric(tracks)
        return_source(
            source_prefix=probes_prefix,
            tracks_reduce=tracks_reduce,
            worker_number=1
        )
        return weight
    if not source_prefix and group_id:
        source_prefix = \
            f'groups/probed/{group_id}'
    obtain_source_tracks(
        source_prefix=source_prefix,
        source_weight=source_weight(),
        source_number=source_number,
        **kwargs
    )

def obtain_source_tracks(
    source_prefix=None,
    source_weight=None,
    source_number=None,
    source_outset=None,
    source_slicer=None,
    tracks_number=None,
    obtain_source=True,
    update_source=None,
    lookup_prefix=None
):
    assert source_prefix, f'{source_prefix=}'
    if lookup_prefix:
        if isinstance(lookup_prefix, str):
            lookup_prefix = [lookup_prefix]
        lookup_prefix = \
            dict.fromkeys(lookup_prefix, False)
    if not isinstance(source_prefix, str):
        source_prefix, *prefix = source_prefix
        if lookup_prefix is None:
            lookup_prefix = {}
        lookup_prefix |= \
            dict.fromkeys(prefix, True)
    if lookup_prefix:
        def return_source_origin():
            return {
                os.path.join(
                    source_prefix,
                    os.path.split(file)[1]
                    ): file
                    for file in locate_source(
                        prefix=lookup_prefix
                    )
            }
        source_origin = return_source_origin()
    def source_entity(source):
        return 'user' \
            if '_' not in str(source) \
                else 'list'
    def return_source(source):
        entity = source_entity(source)
        file = f'{entity}{source}.json'
        return os.path.join(source_prefix, file)
    source_weight = \
        collections.Counter({
            source_id(source): weight
                for source, weight in
                    collections.Counter(
                        source_weight).items()
        })
    if source_slicer is None:
        source_slicer = slice(0, None, 1)
    source_dumped = \
        set(locate_source(prefix=source_prefix))
    if source_outset is True:
        source_outset = \
            next((
                source
                    for source, _ in reversed(
                        source_weight
                            .most_common()
                                [source_slicer]
                    ) if return_source(source)
                        in source_dumped
                ), None
            )
    result_source = []
    for rank, (source, weight) in \
        list(enumerate(
            source_weight.most_common()
            ))[source_slicer]:
        if len(result_source) \
            == source_number:
            break
        if source == source_outset:
            source_outset = None
        file = return_source(source)
        if not update_source:
            if file in source_dumped:
                result_source.append(source)
                continue
        if source_outset:
            continue
        print(source, weight)
        if not lookup_prefix:
            origin_source = None
        else:
            origin_source = \
                source_origin.get(file)
            if origin_source and \
                not os.path.exists(
                    origin_source
                ):
                source_origin = \
                    return_source_origin()
                origin_source = \
                    source_origin.get(file)
        if origin_source:
            target = \
                map(
                    lookup_prefix.get,
                    filter(
                        origin_source.startswith,
                        lookup_prefix
                    )
                )
            if any(target):
                continue
            with open(origin_source, 'r') as f:
                tracks = list(
                    itertools.islice(
                        ijson.items(f, 'item'),
                        tracks_number
                    )
                )
        else:
            tracks = []
        if obtain_source and (
            not origin_source or
                update_source
        ):
            for trial in range(2):
                try:
                    entity = \
                        source_entity(source)
                    if entity == 'user':
                        obtain_tracks = \
                            obtain_user_tracks
                    elif entity == 'list':
                        obtain_tracks = \
                            obtain_list_tracks
                    tracks = \
                        obtain_tracks(
                            source,
                            tracks_number=
                                tracks_number
                            ) or tracks
                    break
                except IOError:
                    if not trial:
                        time.sleep(60)
                    else:
                        raise
        if tracks:
            os.makedirs(
                source_prefix, exist_ok=True
            )
            dump_tracks_file(tracks, file)
            print(
                len(result_source),
                rank,
                len(source_weight),
                file,
                weight,
                len(tracks),
                *(
                    [origin_source]
                        if origin_source
                            else []
                )
            )
            result_source.append(source)




def fetch_tracks(
    tracks,
    folder=None,
    worker_number=None
):
    def worker(id):
        for index, track \
            in enumerate(tracks):
            if index % worker_number != id:
                continue
            file = os.path.join(
                folder or 'tracks',
                    f'{track}.opus'
                        .replace('/', '\\')
            )
            if os.path.exists(file):
                continue
            if fetch_track(track, file):
                print(
                    os.getpid(), index, track
                )
    if not worker_number:
        worker_number = 1
    if worker_number == 1:
        result = list(map(worker, range(1)))
    else:
        try:
            pool = pathos.pools \
                ._ProcessPool(worker_number)
            result = \
                pool.imap_unordered(
                    worker,
                    range(worker_number)
                )
            pool.close()
            pool.join()
        except (Exception, KeyboardInterrupt):
            pool.terminate()
            pool.join()
            raise

def fetch_track(track, file=None):
    def track_id(
        *,
        artist,
        title,
        subtitle=None,
        **kwargs
    ):
        id = f'{artist} - {title}'
        if subtitle:
            id += f'({subtitle})'
        return id.lower()
    tracks = \
        query_tracks(track, full_id=True)
    if not tracks:
        list_page_href, list_uids = \
            query_lists(track)
        if list_page_href:
            list_uids = \
                fetch_lists(list_page_href)
        for list_id in list_uids:
            _tracks = \
                obtain_list_tracks(
                    list_id, full_id=True
                )
            tracks.extend(_tracks)
            _tracks = \
                list(
                    filter(
                        lambda _:
                            track_id(**_)
                                == track,
                        _tracks
                    )
                )
            if _tracks:
                tracks = _tracks
                break
    titles = [
        track_id(**_track)
            for _track in tracks
    ]
    print(f'{titles=}')
    print(f'{track=}')
    counts = \
        collections.Counter(
            list(
                filter(
                    common_tracks(
                        titles, [track]
                        ).__contains__,
                    titles
                )
            ) or titles
        )
    if not counts:
        return None
    naming, _ = counts.most_common(1)[0]
    tracks = [
        _track for _track in tracks if
             track_id(**_track) == naming
    ]
    length = duration(naming)
    if not length:
        track = tracks[0]
    else:
        track = \
            min(
                tracks,
                key=lambda item:
                    abs(
                        item['duration']
                            - length
                    )
            )
    try:
        return fetch_bytes(track, file)
    except Exception:
        traceback.print_exc()
        return fetch_bytes(track, file)

def fetch_bytes(track, file=None):
    if 'url' not in track:
        full_id = track['id']
        assert isinstance(full_id, tuple)
        naming = track_id(**track)
        track = \
            next(
                vk_api.audio.scrap_tracks(
                    [track['id']],
                    client,
                    vk_audio._vk.http
                ), None
            )
        if not track:
            print('!', naming)
            return False
    m3u8_data = \
        m3u8.loads(
            requests.get(
                track['url']).text
        )
    if not m3u8_data.keys:
        return False
    link = m3u8_data.keys[0].uri
    if '/' not in link:
        link = track['url']
    base = link.rsplit('/', 1)[0]
    key = None
    data = b''
    for idx, part in enumerate(
        m3u8_data.segments
    ):
        decrypt = lambda _: _
        if part.key.method == "AES-128":
            if not key:
                link = part.key.uri
                if '/' not in link:
                    link = f'{base}/{link}'
                key = requests \
                    .get(link).content
            seq = idx + m3u8_data \
                .media_sequence
            cipher = aes.new(
                key,
                aes.MODE_CBC,
                iv=binascii.a2b_hex(
                    '%032x' % seq
                )
            )
            decrypt = cipher.decrypt
        link = f'{base}/{part.uri}'
        data += decrypt(
            requests.get(link).content
        )
    if not file:
        file = f'{name}.mp3'
    path, name = os.path.split(file)
    name, _ = os.path.splitext(name)
    temp_file = \
        os.path.join(
            path, f'.{name}.mp3'
        )
    with open(temp_file, 'wb') as f:
        f.write(data)
    copy = file.endswith('.mp3')
    process = \
        subprocess.run((
            'ffmpeg',
            '-loglevel', 'quiet',            
            '-i', temp_file,
            *(('-c', 'copy')
                if copy else ()),
            '-y', file
            ), check=True
        )
    os.remove(temp_file)
    return True

def duration(track):
    return duration_fm(track) \
        or duration_yt(track)

def duration_fm(track):
    import pylast
    network = \
        pylast.LastFMNetwork(
            api_key=api_key,
            api_secret=api_secret,
            username=username,
            password_hash=password_hash
        )
    track = network.get_track(
        track_artist(track)[0],
        track_title(track)
    )
    try:
        return track.get_duration() // 1000
    except:
        return None

def duration_yt(
    track,
    min_views=None,
    max_views=None
):
    title = track.replace('"', '\'')
    audio = '(official audio)'
    query = \
        '|'.join((
            f'"{title} {audio}"',
            f'"{title}"',
            title
        ))
    process = \
        subprocess.run(
            (
                'yt-dlp',
                '--quiet',
                '--no-warnings',
                '--get-duration',
                *(
                    (
                        '--min-views',
                        str(min_views)
                    ) if min_views else ()
                ),
                *(
                    (
                        '--max-views',
                        str(max_views)
                    ) if max_views else ()
                ),
                f'ytsearch1:{query}'
            ),
            check=True,
            stdout=subprocess.PIPE
        )
    if not process.stdout:
        return None
    *mins, secs = \
        process.stdout.decode().split(':')
    mins = mins[0] if mins else '0'
    return int(mins) * 60 + int(secs)
