import builtins
import collections
import datetime
import functools
import glob
import ijson
import itertools
import json
import math
import numpy
import operator
import os
import pathlib
import pathos
import pickle
import random
import regex
import scipy.stats
import sklearn.svm
import string
import subprocess
import sys
import traceback
import typing
import unicodedata




def track_id(
    *,
    artist,
    title,
    subtitle=None,
    collab=True,
    **kwargs
):
    if type(artist) is list:
        artist = ', '.join(artist)
    artist = fixed_artist(artist)
    if not collab:
        artist = track_artist(artist)[0]
    title = fixed_title(title)
    id = f'{artist} - {title}'
    if subtitle:
        subtitle = fixed_string(subtitle)
        if ((
            any(
                map(
                    subtitle.__contains__,
                    tracks_tweaks
                )
            ) and not any(
                map(
                    subtitle.__contains__,
                    tracks_traits
                )
            )) or (any(
                map(
                    subtitle.__contains__,
                    tracks_medias
                )
            ) and any(
                map(
                    title.__contains__,
                    tracks_themes
                )
            )
        )):
            return id + f' ({subtitle})'
    return id

def track_artist(id):
    artist, *_ = id.split(' - ', 1)
    for lexema, titles, spaces \
        in tracks_collab:
        if titles is not True:
            first, other = None, None
            if spaces is not False:
                if f' {lexema} ' in artist:
                    first, other = \
                        artist.split(
                            f' {lexema} ', 1
                        )
                    return [
                        first.strip(),
                        *track_artist(other)
                    ]
            if spaces is not True:
                if lexema in artist:
                    first, other = \
                        artist.split(lexema, 1)
                    return [
                        first.strip(),
                        *track_artist(other)
                    ]
    return [artist.strip()]

def track_title(id):
    return id.split(' - ', 1)[1]

def fixed_artist(artist):
    artist = fixed_string(artist)
    artist = artist.split(' - ')[-1]
    if '(' in artist and ')' in artist:
        artist, _ = artist.split('(', 1)
        artist += _.split(')', 1)[-1]
        artist = fixed_artist(artist)
    return artist.strip()

def fixed_title(title):
    title = fixed_string(title)
    naming, *parens = title.rsplit('(', 1)
    for lexema, titles, spaces in tracks_collab:
        if titles is not False:
            naming = \
                naming.split(f' {lexema} ')[0]
    naming = remove_suffix(naming, ['?', '!'])
    if not parens:
        title = naming
    else:
        parens = parens.pop()
        for lexema in tracks_traits:
            if (f'({lexema})' in f'({parens}' or
                f'({lexema} ' in f'({parens}' or
                f' {lexema} ' in f'({parens}' or
                f' {lexema})' in f'({parens}'):
                if lexema in tracks_medias and \
                    any(
                        map(
                            naming.__contains__,
                            tracks_themes
                        )
                    ):
                    continue
                title = naming
                break
        title = fixed_title(naming) + (
            f' ({parens}'
                if title != naming else ''
        )
    return title.strip()

def fixed_string(string):
    string = ' '.join(string.split())
    string = \
        ''.join(
            filter(
                lambda symbol:
                    unicodedata.category
                        (symbol)[0] != 'C',
                string
            )
        )
    return string.lower() \
        .replace('[', '(') \
        .replace(']', ')') \
        .replace('{', '(') \
        .replace('}', ')') \
        .replace('’', "'")

def remove_suffix(string, suffixes):
    while True:
        _string = string
        for suffix in suffixes:
            string = string \
                .removesuffix(suffix)
        if string == _string:
            break
    return string

def reduce(tracks):
    def action(track):
        return track_id(
            artist=next(iter(
                track_artist(track)
            )),
            title=track_title(track)
        )
    if isinstance(tracks, str):
        return action(tracks)
    else:
        return set(map(action, tracks))

def expand(tracks):
    if isinstance(tracks, str):
        tracks = {tracks}
    tracks = set(tracks)
    for track in list(tracks):
        artist = track_artist(track)
        if not artist:
            continue
        title = track_title(track)
        for collab in (
            ', ',
            ' & ',
            ' and ',
            ' feat. '
        ):
            tracks.add(
                track_id(
                    artist=collab.join(
                        artist[:2]
                    ), title=title
                )
            )
        for _artist in artist:
            tracks.add(
                track_id(
                    artist=_artist,
                    title=title
                )
            )
    return tracks

tracks_collab = (
    ('featuring', None, True),
    ('feat.', None, True),
    ('feat', None, True),
    ('ft.', None, True),
    ('ft', None, True),
    ('with', False, True),
    ('w/', False, True),
    ('and', False, True),
    ('vs.', False, True),
    ('vs', False, True),
    ('pres.', False, True),
    ('pres', False, True),
    ('x', False, True),
    ('&', False, None),
    (',', False, None),
    ('/', False, None),
    ('\\', False, None)
)

tracks_tweaks = (
    'acoustic',
    'orchestra',
    'acapella',
    'stripped',
    'reprise',
    'mashup',
    'flip',
    'bootleg',
    'remix',
    'rmx',
    'vip',
    'rework',
    'remake',
    'cover'
)

tracks_traits = (
    'original',
    'album',
    'radio',
    'club',
    'edit',
    'live',
    'instrumental',
    'dub',
    'minus',
    'slow',
    'slowed',
    'speed',
    'sped',
    'speed-up',
    'reverb',
    'broken',
    'remaster',
    'remastered',
    'feat.',
    'feat',
    'ft.',
    'ft',
    'with',
    'w/',
    'produced',
    'prod.',
    'prod',
    'ost',
    'o.s.t.',
    'from'
)

tracks_medias = (
    'ost',
    'o.s.t.',
    'from'
)

tracks_themes = (
    'theme',
    'title',
    'credits'
)

def intern(tracks):
    return update(tracks, track_ident)

def extern(tracks):
    return update(tracks, ident_track)

def update(tracks, action):
    try:
        tracks = dict(tracks)
    except ValueError:
        pass
    if isinstance(tracks, dict):
        return type(tracks)({
            action(track): value
                for track, value
                    in tracks.items()
        })
    else:
        return type(tracks)(
            map(action, tracks)
        )

def track_ident(track):
    for source, target in \
        tracks_idents.items():
        track = track.replace(
            source, target
        )
    if track[0].isdigit():
        track = '_' + track
    return sys.intern(track)

def ident_track(ident):
    for source, target in \
        tracks_idents.items():
        if target:
            ident = ident.replace(
                target, source
            )
    return ident.lstrip()

tracks_idents = {
    '-': 'X',
    ' ': '_',
    ',': 'O',
    '&': 'A',
    '+': 'P',
    '*': 'M',
    '.': 'D',
    '(': 'I',
    '[': 'I',
    '{': 'I',
    ')': 'J',
    ']': 'J',
    '}': 'J',
    '/': 'F',
    '\\': 'B',
    '|': 'K',
    ':': 'C',
    ';': 'O',
    "'": 'V',
    '"': 'W',
    '«': 'W',
    '»': 'W',
    '!': 'E',
    '?': 'Q',
    '<': 'L',
    '>': 'G',
    '№': 'N',
    '#': 'H',
    '%': 'R',
    '$': 'S'
}

def ignore_tracks():
    return {
        'unknown artist - untitled',
        'неизвестен - без названия'
    }




def load_source_page(
    user=None,
    name=None,
    text=set(),
    date=None,
    flat=True
):
    path = \
        os.path.join(
            'posts',
            f'posts{user or "*"}.json'
        )
    tracks = {
        post['id']
            if user else
                str(post['owner_id']) +
                    '_' + str(post['id']) : _
        for file in glob.iglob(path)
            for post in ijson.items(
                pathlib.Path(file)
                    .read_text(), 'item'
            ) if 'attachments' in post
                and (_ := [
                    track_id(
                        **item['audio'],
                        collab=False
                    ) for item in post
                        ['attachments']
                    if item['type'] == 'audio'
                        and (
                            not name or
                            track_id(
                                **item['audio'],
                                collab=False
                                ) == name
                        )
                    ]
                ) and (
                    not text or any(
                        map(
                            post['text']
                                .__contains__,
                            text
                        )
                    )
                ) and (
                    not date or
                        post['date'] > date
                )
    }
    if flat:
        tracks = \
            list(
                itertools.chain(
                    *tracks.values()
                )
            )
    return tracks

def load_source_kind(id, **kwargs):
    return load_source_file(
        locate_source(
            id, greedy=False
        )[0], **kwargs
    )

load_source_user = load_source_kind
load_source_list = load_source_kind

def load_source_file(
    source,
    filter=None,
    window=None,
    number=None,
    mapper=None,
    pickle=None,
    intern=None
):
    def mapped(tracks):
        if not mapper:
            return tracks
        def sorter(_):
            return mapper(_[0])
        return {
            track: sorted(
                itertools.chain(
                    *map(
                        operator
                            .itemgetter(1),
                        group
                    )
                )
            )
                for track, group in
                    itertools.groupby(
                        sorted(
                            tracks.items(),
                            key=sorter
                        ), key=sorter
                    )
        }
    if pickle:
        tracks = \
            load_source_pickle_file(
                source,
                filter=filter,
                window=window,
                number=number,
                intern=intern
            )
        if tracks is not None:
            return mapped(tracks)
    with open(source, 'r') as f:
        tracks = \
            load_tracks_json(
                itertools.islice(
                    ijson.items(f, 'item'),
                    number
                ),
                filter=filter,
                window=window
            )
        if pickle and (
            filter is None or (
                window == math.inf
                    and tracks
                )
            ) and number is None:
            dump_source_pickle_file(
                tracks, source
            )
        return {
            track_ident(track)
                if intern else
                    track: index
                for track, index in
                    mapped(tracks).items()
        }

def load_tracks_json(
    tracks,
    filter=None,
    window=None
):
    return dict(
        functools.reduce(
            lambda _tracks, track:
                _tracks[track[1]]
                    .append(track[0])
                        or _tracks,
            tracks_picker(
                enumerate(
                    map(
                        lambda track:
                            track_id(
                                **track
                            ),
                        tracks
                    )
                ),
                (
                    lambda item:
                        filter(item[1])
                ) if callable(filter) else
                (
                    lambda item:
                        item[1] in filter
                ) if filter else None,
                window
            ),
            collections.defaultdict(list)
        )
    )

def tracks_picker(
    tracks,
    filter,
    window
):
    if window:
        maxlen = window
        if window == math.inf:
            maxlen = None
        buffer = collections.deque(
            maxlen=maxlen
        )
    tracks = iter(tracks)
    for track in tracks:
        if not filter or filter(track):
            if window:
                yield from buffer
                buffer.clear()
            yield track
            if window:
                yield from \
                    itertools.islice(
                    tracks, maxlen
                )
        else:
            if window:
                buffer.append(track)

def dump_source_pickle_file(
    tracks,
    source,
    sorted=True
):
    source = \
        source.replace(
            '.json', '.pickle'
        )
    with open(source, 'wb') as f:
        pickle.dump(
            tracks
                if not sorted else
            dict(
                __builtins__.sorted(
                    tracks.items()
                )
            ), f
        )

def load_source_pickle_file(
    source,
    filter=None,
    window=None,
    number=None,
    intern=None
):
    source_pickle = \
        source.replace('.json', '.pickle')
    if not os.path.exists(source_pickle):
        return None
    with open(source_pickle, 'rb') as f:
        tracks = pickle.load(f)
        if filter is None and \
            number is None:
            return tracks \
                if not intern else {
                    track_ident(track): index
                        for track, index
                            in tracks.items()
                }
        idxs = set()
        for track in (
            tracks
                if callable(filter)
                    else filter
                        if filter is not None
                            else tracks
        ):
            if (
                filter(track)
                    if callable(filter)
                        else track in tracks
                            if filter is not None
                                else True
                ):
                index = tracks[track]
                if number:
                    if min(index) >= number:
                        continue
                if not window:
                    idxs.update(index)
                else:
                    if window == math.inf:
                        idxs = range(len(tracks))
                        break
                    for idx in index:
                        idxs.update(
                            range(
                                idx - window,
                                idx + window + 1
                            )
                        )
        return {
            track
                if not intern else
                    track_ident(track): index
                for track, index in tracks.items()
                    if (
                        any(
                            builtins.filter(
                                idxs.__contains__,
                                index
                            )
                        )
                            if window != math.inf
                                else idxs
                    ) and (
                        number is None or
                            min(index) < number
                    )
        }




def source_id(source):
    if isinstance(source, int):
        return source
    if isinstance(source, dict):
        return \
            dict(
                zip(
                    map(source_id, source),
                    source.values()
                )
            )
    if not isinstance(source, str):
        return list(map(source_id, source))
    _, source = os.path.split(source)
    if source.endswith('.json'):
        source = source.rsplit('.json', 1)[0]
    if source.startswith('user'):
        return int(source.split('user', 1)[1])
    if source.startswith('list'):
        return source.split('list', 1)[1]
    return source

def unique_source(source):
    def file_name(file):
        if isinstance(file, tuple):
            file = file[0]
        return os.path.split(file)[-1]
    isdict = isinstance(source, dict)
    if isdict:
        source = source.items()
    indexs = {}
    for file in source:
        indexs[file_name(file)] = file
    return (
        dict if isdict else list
        )(indexs.values())

def locate_source(
    source='*',
    prefix='',
    greedy=True
):
    kind = ''
    if source != '*':
        source = source_id(source)
        match source:
            case int(): kind = 'user'
            case str(): kind = 'list'
    name = f'{kind}{source}.json'
    if not greedy and not prefix:
        if os.path.isfile(name):
            return [name]
    if isinstance(prefix, str):
        prefix = [prefix]
    prefix = [
        path
            if path.endswith('.json')
        else os.path.join(
            path or '**', name
            ) for path in prefix
    ]
    result = (
        file for path in prefix
            for file in (
                (path,)
                    if '*' not in path
                else glob.iglob(
                    os.path.join(path),
                    recursive=
                        '**' in path
                )
            )
    )
    if source != '*':
        result = \
            sorted(
                result,
                key=os.path.getsize,
                reverse=True
            )
    return result

def locate_pickle(
    source_prefix,
    tracks_filter=None,
    tracks_weight=None,
    tracks_artist=None
):
    prefix_filter = [
        prefix.replace('.json', '.pickle')
            if prefix.endswith('.json')
        else os.path.join(prefix, '*.pickle')
            for prefix in source_prefix
    ]
    prefix_target = {
        prefix.replace('.json', '.pickle')
            if prefix.endswith('.json')
        else
            os.path.split(
                prefix.split('*')[0]
                )[0] if '*' in prefix
                    else prefix
            for prefix in source_prefix
    }
    _prefix = ''
    for prefix in sorted(prefix_target):
        if prefix.startswith(
            _prefix + os.path.sep):
            prefix_target.remove(prefix)
        else:
            _prefix = prefix
    if not tracks_filter:
        _tracks_filter = [
            '(?-u:\\x94]\\x94)'
        ]
    else:
        _tracks_filter = [
            ('(?-u:\\x8c\\x{:02x})'
                .format(
                    pickle.dumps(track)[12]
                ) if not ord('\n') <=
                    pickle.dumps(track)[12]
                        <= ord('\r')
                else ''
            ) + regex.escape(
                track, True, True
                ) + '(?-u:\\x94)'
            for track in tracks_filter
        ]
    args = (
        'rg',
        *(
            ('--count-matches',)
            if not tracks_filter else
            ('--only-matching',
            '--no-line-number')
        ),
        '--with-filename',
        '--no-heading',
        '--text',
        *itertools.chain(
            *zip(
                itertools.repeat('-e'),
                _tracks_filter
            )
        ),
        *itertools.chain(
            *zip(
                itertools.repeat('-g'),
                prefix_filter
            )
        ),
        *prefix_target
    )
    try:
        proc = \
            subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                text=True,
                errors='ignore'
            )
        source_counts = collections.Counter()
        for line in proc.stdout.readlines():
            source, *_, string = \
                line.strip().split('.pickle:')
            source = source + '.pickle'
            string = string.strip()
            if not tracks_filter:
                source_counts[source] = \
                    int(string)
            else:
                if string[1:] in tracks_filter:
                    string = string[1:]
                if not tracks_artist:
                    source_counts[source] += (
                        1 if not tracks_weight
                        else tracks_weight.get(
                            string, 1
                        )
                    )
                else:
                    artist = \
                        track_artist(string)[0]
                    source_counts.setdefault(
                        source, set()
                        ).add(artist)
        if tracks_artist:
            for source, artist in \
                source_counts.items():
                source_counts[source] \
                    = len(artist)
        return source_counts
    except OSError as e:
        traceback.print_exc()
        return None




def return_source(
    source_tracks=None,
    source_prefix=None,
    source_filter=None,
    source_sorter=None,
    source_merger=None,
    source_number=None,
    source_pickle=None,
    source_search=None,
    tracks_filter=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_common=None,
    tracks_artist=None,
    tracks_window=None,
    tracks_mapper=None,
    tracks_reduce=None,
    tracks_merger=None,
    tracks_number=None,
    tracks_intern=None,
    worker_number=None
):
    if isinstance(source_prefix, str):
        source_prefix = [source_prefix]
    if source_pickle is None:
        source_pickle = True
    if source_search is None:
        source_search = True
    if tracks_filter is not None and \
        not callable(tracks_filter):
        if not isinstance(tracks_filter, set):
            tracks_filter = set(tracks_filter)
        tracks_filter -= ignore_tracks()
    if tracks_intern and \
        isinstance(tracks_filter, set):
            tracks_filter = \
                set(
                    map(
                        lambda track:
                            ident_track(track)
                                if track.
                                    isidentifier()
                                        else track
                        , tracks_filter
                    )
                )
    source = None
    if source_search and \
        tracks_filter and \
        isinstance(tracks_filter, set):
        source = \
            locate_pickle(
                source_prefix,
                tracks_filter,
                tracks_weight,
                tracks_artist
            )
        if not tracks_fscore:
            source = \
                collections.Counter({
                    file.replace(
                        '.pickle', '.json'
                        ): hits
                    for file, hits in
                        source.most_common()
                    if not tracks_common or
                        hits >= tracks_common
                })
        else:
            source_counts = \
                locate_pickle(
                    source
                        if len(source)
                            <= 10000 else
                    source_prefix
                )
            def fscore(file, hits):
                precis = \
                    hits / (
                        source_counts[file]
                            or math.inf
                    )
                recall = \
                    hits / len(tracks_filter)
                weight = tracks_recall or 0.5
                return scipy.stats.hmean(
                    (precis, recall),
                    weights=(
                        1 / weight**2, 1
                    )
                )
            source = \
                collections.Counter({
                    file.replace(
                        '.pickle', '.json'
                    ): round(
                        fscore(file, hits), 6
                    )
                    for file, hits in
                        source.items()
                    if (
                        tracks_fscore is True or
                            fscore(file, hits)
                                >= tracks_fscore
                        ) and (
                        tracks_common is None or
                            hits >= tracks_common
                        )
                })
        if source_sorter is None:
            source = dict(source.most_common())
            source_sorter = False
    if source is None:
        source = \
            locate_source(prefix=source_prefix)
    if source_sorter is None:
        source_sorter = os.path.getmtime
    if source_sorter:
        isdict = isinstance(source, dict)
        if isdict:
            source = source.items()
        source = sorted(source, key=source_sorter)
        if isdict:
            source = dict(source)
    if source_merger is None:
        source_merger = unique_source
    if source_merger:
        source = source_merger(source)
    source_islice = \
        itertools.islice(
            filter(
                lambda file: (
                    source_tracks is None or
                        file not in source_tracks
                ) and (
                    source_filter is None or (
                        source_filter(file)
                            if callable(
                                source_filter
                            ) else
                        file in source_filter
                    )
                ), source
            ), source_number
        )
    if source_tracks or \
        source_filter or \
        source_number:
        source_queued = list(source_islice)
    else:
        source_queued = source
    worker = source_tracks_loader
    params = \
        zip(
            source_queued,
            itertools.repeat((
                tracks_filter,
                tracks_window,
                tracks_number,
                tracks_mapper,
                source_pickle,
                tracks_intern
            )),
            itertools.repeat(
                tracks_reduce
            )
        )
    if not worker_number:
        worker_number = 1
    if worker_number == 1:
        result = map(worker, params)
    else:
        pool = \
            pathos.pools._ProcessPool(
                worker_number
            )
        result = \
            pool.imap_unordered(
                worker, params
            )
    try:
        result = filter(None, result)
        if source_tracks is None:
            source_tracks = {}
        if tracks_reduce is None:
            source_tracks.update(result)
            return {
                file: source_tracks[file]
                    for file in source_queued
                        if file in source_tracks
            }
        if tracks_reduce is True:
            return source_queued
        if callable(tracks_reduce):
            addend = next(result, None)
            if addend is None:
                result = list(result)
                return None
            if isinstance(
                addend, typing.Iterator
            ):
                addend = list(addend)
                result = map(list, result)
            if tracks_merger is None:
                if isinstance(
                    addend, collections.Counter
                ):
                    def merger(c1, c2):
                        for k in c2:
                            if k not in c1:
                                c1[k] = c2[k]
                            else:
                                c1[k] += c2[k]
                        return c1
                elif isinstance(addend, dict):
                    def merger(d1, d2):
                        d1.update(d2)
                        return d1
                else:
                    merger = operator.iadd
            else:
                merger = tracks_merger
            return functools.reduce(
                merger, result, addend
            )
    except (Exception, KeyboardInterrupt):
        if worker_number > 1:
            pool.terminate()
            pool.join()
        raise
    finally:
        if worker_number > 1:
            pool.close()
            pool.join()

def source_tracks_loader(params):
    file, args, func = params
    tracks = load_source_file(file, *args)
    if tracks:
        if func is None:
            return file, tracks
        else:
            return func([(file, tracks)])
    else:
        return None

def period_source_filter(
    before=False,
    **kwargs
):
    timestamp = \
        datetime.datetime(**kwargs)
    if not timestamp:
        return None
    def source_filter(file):
        return before == (
            os.path.getmtime(file)
                < timestamp.timestamp()
        )
    return source_filter

def agesex_source_filter(
    source,
    minage=None,
    maxage=None,
    female=None
):
    if minage or maxage:
        ages = {
            _['id']:
            datetime.datetime.now().year -
                datetime.datetime.strptime(
                    _['bdate'], '%d.%m.%Y'
                    ).year
                for _ in source
                    if 'bdate' in _ and
                        _['bdate']
                            .split('.')[2:]
        }
    if female is not None:
        sexs = {
            _['id']: _['sex'] == 1
                for _ in source
                    if 'sex' in _
        }
    def source_filter(file):
        id = \
            int(
                os.path.split(file)[-1] \
                    .split('.json')[0] \
                        .split('user')[-1]
            )
        if minage and minage > \
            ages.get(id, 0):
            return False
        if maxage and maxage < \
            ages.get(id, math.inf):
            return False
        if female != None and \
            female != sexs.get(id):
            return False
        return True
    return source_filter

def nearby_tracks_filter(
    tracks_metric,
    nearby_source,
    *remote_source
):
    source = [
        load_source_kind(source)
            for source in (
                nearby_source,
                *remote_source
            )
    ]
    def tracks_filter(track):
        counts = tracks_metric(track)
        if not counts:
            return False
        metric = [
            statistics.mean(
                map(
                    operator.itemgetter(0),
                    filter(
                        None,
                        map(
                            counts.get,
                            tracks
                        )
                    )
                )
            ) for tracks in source
        ]
        result = metric[0] / max(metric)
        return result >= 0.95
    return tracks_filter

def labels_tracks_filter(
    tracks_metric,
    tracks_labels,
    target_labels={0}
):
    tracks_counts = \
        collections.Counter(
            itertools.chain(*tracks_labels)
        )
    tracks_labels = {
        track: label
            for label, tracks in
                enumerate(tracks_labels)
            for track in tracks
                if tracks_counts[track] <= 1
    }
    tracks, metric = \
        tracks_metric(
            tracks_labels, counts=False
        )
    tracks_indexs = \
        dict(
            map(
                reversed,
                enumerate(tracks)
            )
        )
    indexs = [
        tracks_indexs[track]
            for track in tracks_labels
                if track in tracks_indexs
    ]
    kernel = metric[:, indexs].copy()
    kernel /= numpy.max(kernel)
    labels = [
        tracks_labels[tracks[index]]
            for index in indexs
    ]
    svmsvc = \
        sklearn.svm.SVC(
            kernel='precomputed'
        )
    svmsvc.fit(kernel, labels)
    def tracks_filter(track):
        index = tracks_indexs.get(track)
        if index is None:
            return False
        kernel = metric[:, [index]].T.copy()
        labels = svmsvc.predict(kernel)
        return labels[0] in target_labels
    return tracks_filter

def artist_tracks_mapper(track):
    artist, title = track.split(' - ', 1)
    artist = track_artist(artist)[0]
    return f'{artist} - {title}'

def common_tracks_reduce(
    tracks_filter,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_metric=None,
    tracks_common=None,
    reduce_artist=None,
    expand_artist=None,
    return_fscore=None,
    return_metric=None,
    return_common=None,
    return_number=None,
    return_tracks=None,
    tracks_reduce=None
):
    if not expand_artist:
        tracks_filter = set(tracks_filter)
    else:
        reduce_artist = None
        tracks_filter = {
            track_artist(track)[0]
                for track in tracks_filter
        }
    if tracks_weight:
        tracks_weight = \
            collections.defaultdict(
                functools.partial(int, 1),
                tracks_weight                
            )
    if not return_fscore and \
        not return_metric and \
        not return_common and \
        not return_number:
        return_tracks = True
    def common_reduce(source_tracks):
        if tracks_reduce:
            source_tracks = \
                tracks_reduce(source_tracks)
        result = []
        for source, tracks in source_tracks:
            if not expand_artist:
                tracks_source = tracks.keys()
            else:
                tracks_source = {
                    track_artist(track)[0]
                        for track in tracks
                }
            common_tracks = \
                tracks_source & tracks_filter
            if reduce_artist:
                common_tracks = {
                    track_artist(track)[0]
                       for track in common_tracks
                }
            if not common_tracks:
                continue
            if not tracks_weight:
                common = len(common_tracks)
            else:
                common = \
                    sum(
                        map(
                            tracks_weight
                                .__getitem__,
                            common_tracks
                        )
                    )
            if tracks_fscore or return_fscore:
                precis = \
                    common / len(tracks_source)
                recall = \
                    common / len(tracks_filter)
                weight = tracks_recall or 0.5
                fscore = \
                    scipy.stats.hmean(
                        (precis, recall),
                        weights=(
                            1 / weight**2, 1
                        )
                    )
                if tracks_fscore and \
                    fscore < tracks_fscore:
                    continue
            if tracks_metric or return_metric:
                log0 = math.log(common)
                log1 = \
                    math.log(len(tracks_source))
                log2 = \
                    math.log(len(tracks_filter))
                logN = math.log(10000)
                metric = \
                    (max(log1, log2) - log0) / \
                        (logN - min(log1, log2))
                metric = math.exp(-metric)
                if tracks_metric and \
                    metric < tracks_metric:
                    continue
            if tracks_common:
                if common < tracks_common:
                    continue
            scores = []
            if return_fscore:
                scores.append(round(fscore, 6))
            if return_metric:
                scores.append(round(metric, 6))
            if return_common:
                scores.append(len(common_tracks))
            if return_number:
                scores.append(len(tracks_source))
            if return_tracks:
                scores.append(tracks)
            if len(scores) > 1:
                scores = (tuple(scores),)
            result.append(
                (source, *scores)
                    if scores else source
            )
        if return_fscore or \
            return_metric or \
            return_common or \
            return_number:
            result = dict(result)
        return result
    return common_reduce

def source_tracks_reduce(
    tracks_reduce=None
):
    def source_reduce(source_tracks):
        if tracks_reduce:
            source_tracks = \
                tracks_reduce(source_tracks)
        source_counts = collections.Counter()
        for source, tracks in source_tracks:
            source_counts[source] = len(tracks)
        return source_counts
    return source_reduce

def counts_tracks_reduce(
    tracks_reduce=None
):
    def counts_reduce(source_tracks):
        if tracks_reduce:
            source_tracks = \
                tracks_reduce(source_tracks)
        tracks_counts = collections.Counter()
        for source, tracks in source_tracks:
            tracks_counts.update(iter(tracks))
        return tracks_counts
    return counts_reduce

def impact_tracks_reduce(
    tracks_filter,
    tracks_reduce=None
):
    if not isinstance(tracks_filter, set):
        tracks_filter = set(tracks_filter)
    def impact_reduce(source_tracks):
        if tracks_reduce:
            source_tracks = \
                tracks_reduce(source_tracks)
        tracks_impact = collections.Counter()
        for source, tracks in source_tracks:
            tracks_common = \
                tracks.keys() & tracks_filter
            tracks_weight = \
                collections.Counter(
                    dict.fromkeys(
                        tracks_common,
                        1 / len(tracks_common)
                    )
                )
            for track in tracks:
                if track not in tracks_filter:
                    if track not in tracks_impact:
                        tracks_impact[track] = \
                            collections.Counter()
                    tracks_impact[track] += \
                        tracks_weight
        return tracks_impact
    return impact_reduce

def proxim_tracks_reduce(
    tracks_filter,
    tracks_weight=None,
    tracks_artist=None,
    tracks_window=1,
    tracks_proxim=None,
    tracks_reduce=None
):
    if not isinstance(tracks_filter, set):
        tracks_filter = set(tracks_filter)
    if not tracks_weight:
        tracks_weight = \
            dict.fromkeys(tracks_filter, 1)
    if tracks_artist:
        tracks_artist = {
            track: track_artist(track)[0]
                for track in tracks_filter
        }
        artist_counts = \
            collections.Counter(
                tracks_artist.values()
            )
    def proxim_reduce(source_tracks):
        if tracks_reduce:
            source_tracks = \
                tracks_reduce(source_tracks)
        tracks_counts = collections.Counter()
        for source, tracks in source_tracks:
            common_tracks = \
                tracks.keys() & tracks_filter
            if not common_tracks:
                continue
            indexs = {}
            for track in common_tracks:
                weight = \
                    tracks_weight.get(track, 1) \
                        / len(tracks[track])
                if tracks_artist:
                    weight /= \
                        artist_counts[
                            tracks_artist[track]
                        ]
                for index in tracks[track]:
                    for shift in range(
                        -tracks_window,
                        tracks_window + 1
                    ):
                        if tracks_proxim:
                            proxim = 1 - (
                                abs(shift or 1)-1
                                ) / tracks_window
                        else:
                            proxim = 1
                        indexs[index + shift] = \
                            max(
                                indexs.setdefault(
                                    index + shift,
                                    0
                                ), proxim * weight
                            )
            for track in tracks:
                weight = 0
                for _index in tracks[track]:
                    _weight = indexs.get(_index)
                    if _weight:
                        weight = \
                            max(_weight, weight)
                if weight:
                    tracks_counts[track] += weight
        for track, count in \
            tracks_counts.items():
            tracks_counts[track] \
                = round(count, 3)
        return tracks_counts
    return proxim_reduce

def filter_tracks(
    tracks,
    string,
    action=str.__contains__,
    negate=False
):
    def filter_entity(entity):
        if isinstance(entity, tuple):
            entity = entity[0]
        if isinstance(string, str):
            return not negate == \
                action(entity, string)
        else:
            return negate == (
                not next(
                    filter(
                        lambda _:
                            action(entity, _),
                        string
                    ), False
                )
            )
    if isinstance(tracks, dict):
        return type(tracks)(
            dict(
                filter_tracks(
                    list(tracks.items()),
                    string,
                    action,
                    negate
                )
            )
        )
    else:
        return type(tracks)(
            filter(
                filter_entity, tracks
            )
        )




def user2item(
    source_prefix=None,
    source_filter=None,
    source_number=None,
    tracks_filter=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_common=None,
    tracks_artist=None,
    tracks_window=None,
    tracks_proxim=None,
    tracks_debias=None,
    tracks_mapper=None,
    worker_number=None
):
    if source_number:
        if not tracks_fscore and \
            not tracks_common:
            tracks_fscore = True
    if tracks_filter:
        if not tracks_window:
            tracks_window = math.inf
    args = dict(
        source_prefix=source_prefix,
        source_filter=source_filter,
        source_number=source_number,
        tracks_filter=tracks_filter,
        tracks_weight=tracks_weight,
        tracks_fscore=tracks_fscore,
        tracks_recall=tracks_recall,
        tracks_common=tracks_common,
        tracks_artist=tracks_artist,
        tracks_mapper=tracks_mapper,
        worker_number=worker_number
    )
    if not tracks_debias:
        tracks_counts = {}
    else:
        tracks_counts = \
            return_source(
                tracks_reduce=
                    counts_tracks_reduce(),
                **args
            )
        tracks_counts = {
            track: 1 / (1 +
                tracks_debias *
                    math.log(count)
                ) for track, count in
                    tracks_counts.items()
        }
    if not tracks_weight and \
        not tracks_artist and \
        not tracks_proxim and \
        not tracks_debias:
        tracks_reduce = \
            counts_tracks_reduce()
    else:
        tracks_reduce = \
            proxim_tracks_reduce(
                tracks_filter,
                tracks_counts,
                tracks_artist,
                tracks_window,
                tracks_proxim=True
            )
    return return_source(
        tracks_window=tracks_window,
        tracks_reduce=tracks_reduce,
        **args
    )




def item2item(
    caches_suffix,
    cached_metric=None,
    update_metric=None,
    source_prefix=None,
    source_filter=None,
    source_number=None,
    tracks_filter=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_common=None,
    tracks_window=None,
    tracks_mapper=None,
    tracks_reduce=None,
    tracks_number=None,
    tracks_offset=None,
    lookup_transp=None,
    worker_number=None
):
    def return_counts(update=False):
        source_cached = \
            'source_counts.{}.pickle' \
                .format(caches_suffix)
        if os.path.exists(source_cached):
            with open(source_cached, 'rb') as f:
                source_counts = pickle.load(f)
        else:
            source_counts = {}
        tracks_cached = \
            'tracks_counts.{}.pickle' \
                .format(caches_suffix)
        if os.path.exists(tracks_cached):
            with open(tracks_cached, 'rb') as f:
                tracks_counts = pickle.load(f)
        else:
            tracks_counts = {}
        if not update:
            assert source_counts, source_cached
            assert tracks_counts, tracks_cached
        if source_counts and tracks_counts:
            return source_counts, tracks_counts
        assert source_prefix, f'{source_prefix=}'
        source_counts = {}
        tracks_counts = collections.Counter()
        def _tracks_reduce(source_tracks):
            if tracks_reduce:
                source_tracks = \
                    tracks_reduce(source_tracks)
            for source, tracks in source_tracks:
                source_counts[source] = len(tracks)
                tracks_counts.update(iter(tracks))
        return_source(
            source_prefix=source_prefix,
            source_filter=source_filter,
            source_number=source_number,
            tracks_filter=tracks_filter,
            tracks_weight=tracks_weight,
            tracks_fscore=tracks_fscore,
            tracks_common=tracks_common,
            tracks_window=tracks_window,
            tracks_mapper=tracks_mapper,
            tracks_reduce=_tracks_reduce,
            worker_number=1
        )
        if tracks_number:
            tracks_counts = \
                dict(
                    tracks_counts.most_common(
                        tracks_number
                    )
                )
        with open(source_cached, 'wb') as f:
            pickle.dump(source_counts, f)
        with open(tracks_cached, 'wb') as f:
            pickle.dump(tracks_counts, f)
        return source_counts, tracks_counts
    def return_indexs(update=False):
        indexs_cached = \
            'tracks_source.{}.pickle' \
                .format(caches_suffix)
        if not update or \
            os.path.exists(indexs_cached):
            with open(indexs_cached, 'rb') as f:
                return pickle.load(f)
        assert source_prefix, f'{source_prefix=}'
        source_counts, tracks_counts = \
            return_counts()
        source_ranked = \
            dict(
                map(
                    reversed,
                    enumerate(source_counts)
                )
            )
        tracks_source = {}
        def _tracks_reduce(source_tracks):
            def _tracks_reduce(source_tracks):
                if tracks_reduce:
                    source_tracks = \
                        tracks_reduce(
                            source_tracks
                        )
            for source, tracks in source_tracks:
                if source not in source_ranked:
                    continue
                number = source_ranked[source]
                for track in tracks:
                    if track in tracks_counts:
                        tracks_source. \
                            setdefault(
                                track, set()
                                ).add(number)
        return_source(
            source_prefix=source_counts,
            source_search=False,
            tracks_filter=tracks_counts
                if not tracks_mapper else (
                    lambda track:
                        tracks_mapper(track)
                            in tracks_counts
                ),
            tracks_mapper=tracks_mapper,
            tracks_reduce=_tracks_reduce,
            worker_number=1
        )
        tracks_source = [
            tracks_source.get(track, set())
                for track in tracks_counts
        ]
        with open(indexs_cached, 'wb') as f:
            pickle.dump(tracks_source, f)
        return tracks_source
    def obtain_metric(
        index1,
        source_weight,
        source_counts,
        tracks_counts,
        tracks_source,
        metric,
        common,
        normed=True,
        invert=True,
        power0=1,
        power1=0,
        power2=0
    ):
        metric_vector = \
            numpy.zeros(
                shape=(len(tracks_counts),),
                dtype='float16'
            )
        common_vector = \
            numpy.zeros(
                shape=(len(tracks_counts),),
                dtype='float16'
            )
        track1, count1 = \
            next(
                itertools.islice(
                    tracks_counts.items(),
                    index1,
                    None
                )
            )
        if not source_weight:
            if not tracks_source[index1]:
                print(f'! {track1=}')
                return metric_vector, \
                    common_vector
        else:
            if not isinstance(
                source_weight, dict):
                source_weight = \
                    dict.fromkeys(
                        source_weight, 1
                    )
            source_ranked = \
                dict(
                    map(
                        reversed,
                        enumerate(
                            source_counts
                        )
                    )
                )
            source_weight = {
                source_ranked.get(
                    source, source): weight
                    for source, weight in
                        source_weight.items()
            }
            def return_source_counts(source):
                counts = \
                    filter(
                        None, map(
                            source_weight.get,
                            source
                        )
                    )
                if power0 != 1:
                    counts = (
                        count ** power0
                            for count in counts
                    )
                return sum(counts)
        if source_weight and normed:
            count1 = \
                return_source_counts(
                    tracks_source[index1]
                )
            if not count1:
                return metric_vector, \
                    common_vector
        if metric is not None:
            print(os.getpid(), index1, track1)
        for index2, (track2, count2) in \
            enumerate(tracks_counts.items()):
            if index1 == index2:
                continue
            if metric is not None:
                if metric[index1][index2]:
                    continue
                if lookup_transp and \
                    index1 > index2 and \
                    metric[index2][index1]:
                    metric_vector[index2] = \
                        metric[index2][index1]
                    common_vector[index2] = \
                        common[index2][index1]
                continue
            if not tracks_source[index2]:
                continue
            if source_weight and normed:
                count2 = \
                    return_source_counts(
                        tracks_source[index2]
                    )
                if not count2:
                    continue
            source_common = \
                tracks_source[index1] \
                    & tracks_source[index2]
            if not source_common:
                continue
            if not source_weight:
                count0 = len(source_common)
            else:
                count0 = \
                    return_source_counts(
                        source_common
                    )
                if not count0:
                    continue
            countN = len(source_counts)
            log0 = math.log(count0, 2)
            log1 = math.log(count1, 2)
            log2 = math.log(count2, 2)
            logN = math.log(countN, 2)
            result = \
                (max(log1, log2) - log0) \
                    / (logN - min(log1, log2))
            if invert:
                result = 1 / result \
                    if result else math.inf
            metric_vector[index2] = \
                round(result, 6)
            _result = count0 \
                / count1 ** power1 \
                    / count2 ** power2
            if not invert:
                _result = 1 / _result
            common_vector[index2] = \
                round(_result, 6)
        return metric_vector, common_vector
    def worker(number):
        source_counts, tracks_counts = \
            return_counts()
        tracks_source = \
            return_indexs()
        metric, common = \
            metric_common(mode='r+')
        for index1 in range(
            tracks_offset or 0,
            len(tracks_counts)
        ):
            index = index1 % worker_number
            if index == number:
                metric_vector, \
                    common_vector = \
                        obtain_metric(
                            index1,
                            dict(),
                            source_counts,
                            tracks_counts,
                            tracks_source,
                            metric,
                            common
                        )
                metric.flush()
                common.flush()
    def obtain_tracks_metric():
        if worker_number == 1:
            result = map(worker, range(1))
        else:
            pool = \
                pathos.pools._ProcessPool(
                    worker_number
                )
            result = \
                pool.imap_unordered(
                    worker,
                    range(worker_number)
                )
        try:
            _ = list(result)
        except (Exception, KeyboardInterrupt):
            if worker_number > 1:
                pool.terminate()
                pool.join()
            raise
        finally:
            if worker_number > 1:
                pool.close()
                pool.join()
    def metric_common(mode, prefix=None):
        metric_cached = \
            f'metric.{caches_suffix}.memmap'
        if prefix:
            metric_cached = \
                prefix + metric_cached
        if not os.path.exists(metric_cached):
            os.mknod(metric_cached)
        metric = numpy.memmap(
            metric_cached,
            mode=mode,
            shape=(
                len(tracks_counts),
                len(tracks_counts)
            ),
            dtype='float16'
        )
        common_cached = \
            f'common.{caches_suffix}.memmap'
        if not os.path.exists(common_cached):
            if mode == 'r':
                return metric, None
            else:
                os.mknod(common_cached)
        common = numpy.memmap(
            common_cached,
            mode=mode,
            shape=(
                len(tracks_counts),
                len(tracks_counts)
            ),
            dtype='float16'
        )
        return metric, common
    if not worker_number:
        worker_number = 1
    source_counts, tracks_counts = \
        return_counts(update=True)
    ranked_source = list(source_counts)
    source_ranked = \
        dict(
            map(
                reversed,
                enumerate(source_counts)
            )
        )
    ranked_tracks = list(tracks_counts)
    tracks_ranked = \
        dict(
            map(
                reversed,
                enumerate(tracks_counts)
            )
        )
    tracks_source = \
        return_indexs(update=True)
    indexs_artist = {}
    source_indexs = {}
    if cached_metric:
        if update_metric:
            obtain_tracks_metric()
        _metric, _common = \
            metric_common(mode='r')
    else:
        _metric, _common = None, None
    def tracks_metric(
        track1,
        source=None,
        ranked=None,
        metric=None,
        common=None,
        counts=True,
        normed=True,
        invert=True,
        power0=1,
        power1=0,
        power2=0
    ):
        if isinstance(track1, str):
            index1 = tracks_ranked.get(track1)
            if index1 is None:
                return None
        else:
            index1 = [
                tracks_ranked[track]
                    for track in track1
                        if track in tracks_ranked
            ]
        if _metric is not None:
            metric_vector = _metric[index1]
            if _common is not None:
                common_vector = _common[index1]
            else:
                common_vector = numpy.zeros(
                    shape=(len(tracks_counts),),
                    dtype='float16'
                )
        else:
            metric_vector, common_vector = \
                obtain_metric(
                    index1,
                    source_weight=source,
                    source_counts=source_counts,
                    tracks_counts=tracks_counts,
                    tracks_source=tracks_source,
                    metric=_metric,
                    common=_common,
                    normed=normed,
                    invert=invert,
                    power0=power0,
                    power1=power1,
                    power2=power2
                )
        if not counts:
            return ranked_tracks, metric_vector
        if ranked:
            def ranker(vector, output):
                indexs = vector.argsort()
                output[indexs] = \
                    numpy.arange(
                        1, len(vector) + 1
                    ) / len(vector)
            matrix = \
                numpy.zeros(
                    shape=(
                        2, len(tracks_counts)
                    ), dtype=float
                )
            if metric is not False:
                ranker(metric_vector, matrix[0])
                if common is False:
                    ranked_vector = matrix[0]
            if common is not False:
                ranker(common_vector, matrix[1])
                if metric is False:
                    ranked_vector = matrix[1]
            if metric is not False and \
                common is not False:
                ranked_vector = \
                    scipy.stats.hmean(matrix)
        else:
            ranked_vector = itertools.repeat(0)
        return collections.Counter(
            dict((
                track2,
                (
                    *(
                        (track2_ranked,)
                            if ranked else ()
                    ),
                    *(
                    (track2_metric, track2_common)
                        if metric or not common
                        else
                    (track2_common, track2_metric)
                    )
                ))
                for track2,
                    track2_metric,
                    track2_common,
                    track2_ranked in zip(
                        tracks_counts,
                        metric_vector,
                        common_vector,
                        ranked_vector
                    )
                if track2_metric and (
                    type(metric) != int or
                        track2_metric >= metric
                ) and (
                    type(common) != int or
                        track2_common >= common
                )
            )
        )
    return tracks_metric




def tracks_scores(
    tracks_seeded,
    source_prefix=None,
    source_number=None,
    source_filter=None,
    source_weight=None,
    metric_weight=None,
    tracks_metric=None,
    tracks_number=None,
    tracks_filter=None,
    tracks_mapper=None,
    tracks_cutoff=None,
    tracks_ranked=None,
    metric_ranked=None,
    common_ranked=None,
    worker_number=None
):
    if source_filter:
        if not callable(source_filter):
            source_filter = \
                source_filter.__contains__
    if tracks_filter:
        if not callable(tracks_filter):
            tracks_filter = \
                tracks_filter.__contains__
    if not tracks_metric:
        scores = \
            user2item(
                source_prefix=source_prefix,
                source_filter=source_filter,
                source_number=source_number,
                tracks_filter=tracks_filter,
                tracks_mapper=tracks_mapper,
                worker_number=worker_number
            )
        for track in tracks_seeded:
            del scores[track]
    else:
        tracks_scores = {}
        for track in tracks_seeded:
            metric = \
                tracks_metric(
                    track,
                    source=
                        source_weight
                            if metric_weight
                                else None,
                    ranked=tracks_ranked,
                    metric=metric_ranked,
                    common=common_ranked,
                    normed=True,
                    power0=2,
                    power1=0.5,
                    power2=0.5
                )
            if metric is None:
                continue
            def metric_filter(cursor):
                track = cursor[0]
                return (
                    track not in tracks_seeded
                    and (
                        not tracks_filter or
                            tracks_filter(track)
                    )
                )
            tracks_scores[track] = \
                list(
                    itertools.islice(
                        filter(
                            metric_filter,
                            metric.most_common(
                                tracks_cutoff
                            )
                        ), tracks_number
                    )
                )
            print(
                track,
                tracks_scores[track][:1]
            )
        if not source_weight:
            scores = collections.Counter()
            for metric in \
                itertools.zip_longest(
                    *tracks_scores.values()
                ):
                for cursor in metric:
                    if cursor is not None:
                        scores[cursor[0]] += \
                            cursor[1][0]
                if tracks_cutoff and \
                    len(scores) >= tracks_cutoff:
                    break
        else:
            tracks_scored = \
                set(
                    itertools.chain(
                        *map(
                            dict,
                            tracks_scores.values()
                        )
                    )
                )
            scores = \
                user2item(
                    source_prefix=source_prefix,
                    source_filter=source_filter,
                    source_number=source_number,
                    tracks_filter=tracks_scored,
                    tracks_mapper=tracks_mapper,
                    worker_number=worker_number
                )
    for track in scores:
        scores[track] = \
            round(scores[track], 6)
    return scores

def assess_tracks(
    tracks,
    seeded,
    keeped=None,
    scored=None,
    random=None,
    series=None,
    moment=None,
    nearby=None,
    filter=None,
    number=None,
    artist=None,
    prefix=None,
    tracks_metric=None
):
    if not isinstance(tracks, list):
        tracks = list(tracks)
    if keeped is None:
        keeped = len(tracks) - seeded
    if isinstance(keeped, tuple):
        if not moment:
            keeped = keeped[1]
        else:
            from random import randint
            keeped = randint(
                keeped[0], keeped[1]
            )
    if scored is None:
        scored = keeped
    if random:
        if not series:
            from random import shuffle
            shuffle(tracks)
        else:
            outset = len(tracks) \
                - (seeded + keeped)
            from random import randint
            del tracks[:randint(0, outset)]
    tracks = tracks[:seeded + keeped]
    tracks_seeded = set(tracks[:seeded])
    tracks_keeped = set(tracks[seeded:])
    tracks_scored = \
        tracks_scores(
            tracks_seeded,
            source_prefix=prefix,
            source_number=nearby,
            source_filter=filter,
            source_weight=True,
            metric_weight=False,
            tracks_metric=tracks_metric,
            tracks_number=number,
            tracks_ranked=None,
            metric_ranked=True,
            common_ranked=None
        )
    tracks_precis = \
        set(
            dict(
                tracks_scored
                    .most_common(
                        len(tracks_keeped)
                    )
            )
        )
    tracks_common = \
        tracks_precis & tracks_keeped
    prec = len(tracks_common)
    if artist:
        def tracks_artist(track):
            return track.split(' - ', 1)[0]
        artist_keeped = \
            set(
                map(
                    tracks_artist,
                    tracks_keeped
                )
            )
        artist_precis = \
            set(
                map(
                    tracks_artist,
                    tracks_precis -
                        tracks_keeped
                )
            )
        artist_common = \
            artist_precis & artist_keeped
        prec += 0.25 * len(artist_common)
    if prec:
        prec /= len(tracks_keeped)
    tracks_dcgain = \
        tracks_scored.most_common(scored)
    ndcg, idcg = 0, 0
    for index, (track, score) \
        in enumerate(tracks_dcgain):
        if track in tracks_keeped:
            ndcg += 1 / math.log(index + 2, 2)
        idcg += 1 / math.log(index + 2, 2)
    ndcg /= idcg
    return (
        prec,
        ndcg,
        len(tracks_seeded),
        len(tracks_keeped)
    )

def assess_source(
    source,
    slicer=None,
    seeded=None,
    keeped=None,
    greedy=None,
    scored=None,
    random=None,
    series=None,
    moment=None,
    nearby=None,
    number=None,
    artist=None,
    prefix=None,
    metric=None
):
    if isinstance(source, str):
        source = {source: 1}
    if not isinstance(source, dict):
        source = dict.fromkeys(source, 1)
    result = []
    for file, name in source.items():
        tracks = load_source_file(file)
        if len(tracks) < (
            seeded + (
                (
                    keeped[0] + (
                        keeped[1] - keeped[0]
                        ) * float(greedy or 0)
                )
                if type(keeped) == tuple else
                (
                    (keeped or 1)
                        * float(greedy or 1)
                )
            )
        ):
            continue
        if slicer:
            if slicer.stop is not None:
                if slicer.stop <= len(result):
                    break
            if slicer.start is not None:
                if slicer.start > 0:
                    slicer = slice(
                        slicer.start-1,
                        slicer.stop-1
                            if slicer.stop
                                else None
                    )
                    continue
        tracks = list(map(curate, tracks))
        filter = lambda _: _ != file
        prec, ndcg, _seeded, _keeped = \
            assess_tracks(
                tracks=tracks,
                seeded=seeded,
                keeped=keeped,
                scored=scored,
                random=random,
                series=series,
                moment=moment,
                nearby=nearby,
                filter=filter,
                number=number,
                artist=artist,
                prefix=prefix,
                metric=metric
            )
        result.append((prec, ndcg))
        _prec, _ndcg = \
            numpy.mean(result, axis=0)
        print(
            len(result)-1,
            file,
            len(tracks),
            _seeded,
            _keeped,
            (round(prec, 3),
            round(ndcg, 3)),
            (round(_prec, 3),
            round(_ndcg, 3)),
            name
        )
    return numpy.mean(result, axis=0)

def assess(
    source=None,
    seeded=None,
    keeped=None,
    moment=None,
    nearby=100,
    number=100,
    sample=1,
    prefix=None,
    metric=None
):
    return numpy.mean([
        assess_source(
            source=source,
            seeded=seeded,
            keeped=keeped,
            scored=500,
            random=True,
            series=False,
            moment=moment,
            nearby=nearby,
            number=number,
            artist=True,
            prefix=prefix,
            metric=metric
        ) for _ in range(sample)
        ], axis=0
    )
