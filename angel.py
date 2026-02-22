import builtins
import collections
import datetime
import functools
import glob
import ijson
import itertools
import math
import multiprocessing
import numpy
import operator
import os
import pickle
import regex
import scipy.stats
import sklearn.svm
import subprocess
import sys
import traceback
import typing
import unicodedata




def return_source(
    source_prefix=None,
    source_filter=None,
    source_sorter=None,
    source_merger=None,
    source_number=None,
    source_pickle=None,
    source_search=None,
    source_indexs=None,
    source_reduce=None,
    tracks_filter=None,
    tracks_window=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_common=None,
    tracks_artist=None,
    tracks_mapper=None,
    tracks_number=None,
    tracks_reduce=None,
    tracks_merger=None,
    thread_number=None
):
    if isinstance(source_prefix, str):
        source_prefix = [source_prefix]
    if source_pickle is None:
        source_pickle = True
    if source_search is None:
        source_search = True
    if source_indexs is not None:
        if source_indexs == True:
            source_indexs = {}
        assert isinstance(source_indexs, dict)
    if tracks_filter and \
        not callable(tracks_filter):
        if not isinstance(tracks_filter, set):
            tracks_filter = set(tracks_filter)
        tracks_filter = \
            expand_artist(tracks_filter)
        tracks_filter -= ignore_tracks()
        if tracks_fscore is None:
            tracks_fscore = True
    if tracks_filter:
        if tracks_window is None:
            tracks_window = math.inf
    if tracks_artist is None:
        tracks_artist = True
    if tracks_reduce is None:
        tracks_reduce = True
    source = None
    if source_indexs is not None and \
        tracks_filter and \
        not callable(tracks_filter):
        artist_filter = \
            set(itertools.chain(*map(
                track_artist, tracks_filter)))
        source = collections.Counter()
        for artist in artist_filter:
            indexs = source_indexs.get(artist)
            if indexs is None:
                indexs = \
                    source_indexs.setdefault(
                        artist, dict())
                string = \
                    os.path.join(
                        'artist', artist, '*')
                for entity in glob.glob(string):
                    _, _entity = \
                        os.path.split(entity)
                    _entity, _ = \
                        os.path.splitext(_entity)
                    _entity = int(_entity)
                    with open(entity, 'rb') as f:
                        indexs.update(
                            zip(pickle.load(f),
                                itertools.repeat(
                                    _entity)))
            source.update(iter(indexs))
        indexs_number = 2500
        prefix = \
            set(map(lambda _:
                os.path.split(_)[0],
                source_prefix))
        source = \
            collections.Counter({
                file: counts
                for id, counts in
                    source.most_common(
                        indexs_number)
                for artist, indexs in
                    source_indexs.items()
                    if artist in artist_filter
                        and id in indexs
                for _prefix in prefix
                    if (file :=
                        os.path.join(
                            _prefix,
                            str(indexs[id]),
                            f'{id}.json')) and
                        (len(prefix) == 1 or
                        os.path.isfile(file))})
    if source_search and \
        source_pickle and \
        tracks_filter and \
        not callable(tracks_filter):
        source = \
            collections.Counter({
                _[0].removesuffix(
                    '.pickle') + '.json':
                        round(_[1], 3)
                for _ in locate_pickle(
                    source or source_prefix,
                    tracks_filter,
                    tracks_weight,
                    tracks_artist).items()
                if tracks_common is None or
                    _[1] >= tracks_common})
        if tracks_fscore:
            _source = \
                locate_pickle(
                    source, return_cmdarg=True)
            _tracks = \
                locate_pickle(
                    [], tracks_filter,
                        return_cmdarg=True)
            bundle = \
                int(len(source) * 0.7 / (
                    len(' '.join(_source)) / (
                    os.sysconf('SC_ARG_MAX')
                        - len(' '.join(_tracks))
                    ))) + 1
            counts = \
                sum(map(
                    locate_pickle,
                    itertools.batched(
                        source, bundle
                    )), collections.Counter())
            def _fscore(_source, common):
                _source = \
                    _source.removesuffix(
                        '.json') + '.pickle'
                precis = common / \
                    (counts[_source] or math.inf)
                recall = common / \
                    len(tracks_filter)
                return fscore(
                    precis, recall, tracks_recall
                )
            source = \
                collections.Counter({
                    _[0]: round(_fscore(*_), 3)
                    for _ in source.items()
                    if tracks_fscore is True or
                        _fscore(*_)>=tracks_fscore
                    })
        if source_sorter is None:
            source = \
                collections.Counter(
                    dict(source.most_common()))
            source_sorter = False
    if source is None:
        source = \
            collections.Counter(
                locate_source(
                    prefix=source_prefix))
    if source_sorter is None:
        source_sorter = os.path.getmtime
    if source_sorter:
        source = \
            collections.Counter(dict(
                sorted(
                    source.items(), key=lambda _:
                    (source_sorter(_[0]), _[1]))))
    if source_merger is None:
        source_merger = unique_source
    if source_merger:
        source = source_merger(source)
    if source_filter or source_number:
        source_islice = \
            itertools.islice(
                filter(
                    lambda _:
                        source_filter is None or (
                            source_filter(_[0])
                                if callable(
                                    source_filter
                                ) else
                            _[0] in source_filter
                        ), source.items()
                    ), source_number)
        source_queued = \
            collections.Counter(
                dict(source_islice))
    else:
        source_queued = source
    def worker(_):
        _source, _params, _reduce = _
        try:
            tracks = \
                import_source(_source, *_params)
        except Exception:
            print(_source)
            raise
        if tracks:
            if not _reduce:
                return _source, tracks
            else:
                return _reduce([(_source, tracks)])
        else:
            return None
    params = \
        zip(source_queued,
            itertools.repeat((
                tracks_filter,
                tracks_window,
                tracks_number,
                tracks_mapper,
                source_pickle)),
            itertools.repeat(tracks_reduce))
    if not thread_number:
        thread_number = 1
    if thread_number == 1:
        result = map(worker, params)
    else:
        thread = \
            multiprocessing.pool.ThreadPool(
                thread_number)
        result = \
            thread.imap_unordered(worker, params)
    try:
        result = filter(None, result)
        if not tracks_reduce:
            result = dict(result)
        if tracks_reduce is True:
            return source_queued
        if callable(tracks_reduce):
            addend = next(result, None)
            if addend is None:
                result = list(result)
                return None
            if isinstance(
                addend, typing.Iterator):
                addend = list(addend)
                result = map(list, result)
            if tracks_merger is None:
                if isinstance(
                    addend, collections.Counter):
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
            result = \
                functools.reduce(
                    merger, result, addend)
        if callable(source_reduce):
            result = source_reduce(result.items())
        if thread_number > 1:
            worker.close()
            worker.join()
        return result
    except Exception, KeyboardInterrupt:
        if thread_number > 1:
            worker.terminate()
            worker.join()
        raise

def period_source_filter(before=False, **kwargs):
    period = datetime.datetime(**kwargs)
    if not period: return None
    def source_filter(_):
        return before == (
            os.path.getmtime(_)
                < period.timestamp())
    return source_filter

def agesex_source_filter(
    source,
    minage=None,
    maxage=None,
    gender=None
):
    if minage or maxage:
        date = datetime.datetime.now()
        ages = {
            _['id']: date.year -
                datetime.datetime.strptime(
                    _['bdate'], '%d.%m.%Y').year
            for _ in source
                if 'bdate' in _ and
                    _['bdate'].split('.')[2:]}
    if gender:
        sexs = {
            _['id']: _['sex']
                for _ in source if 'sex' in _}
    def source_filter(file):
        id = source_id(file)
        if minage and minage > \
            ages.get(id, 0): return False
        if maxage and maxage < \
            ages.get(id, math.inf): return False
        if gender and gender != \
            sexs.get(id): return False
        return True
    return source_filter

def nearby_tracks_filter(
    tracks_metric,
    nearby_source,
    *remote_source,
    nearby_cutoff=0.95
):
    source = [
        return_tracks(_) for _ in
            (nearby_source, *remote_source)]
    def tracks_filter(track):
        metric = tracks_metric(track)
        if not metric: return False
        def scores(tracks):
            return map(
                lambda _:
                _[0] if type(_) is tuple else _,
                filter(None, map(metric, tracks)))
        _scores = [
            statistics.mean(scores(tracks))
                for tracks in source]
        return _scores[0] / max(_scores) \
            >= nearby_cutoff
    return tracks_filter

def labels_tracks_filter(
    tracks_metric,
    tracks_labels,
    target_labels={0}
):
    tracks_counts = \
        collections.Counter(
            itertools.chain(*tracks_labels))
    tracks_labels = {
        track: label
            for label, tracks in
                enumerate(tracks_labels)
            for track in tracks
                if tracks_counts[track] <= 1}
    metric = \
        tracks_metric(tracks_labels, counts=False)
    tracks = list(metric)
    tracks_indexs = \
        dict(map(reversed, enumerate(tracks)))
    indexs = [
        tracks_indexs[track]
            for track in tracks_labels
                if track in tracks_indexs]
    kernel = metric[:, indexs].copy()
    kernel /= numpy.max(kernel)
    labels = [
        tracks_labels[tracks[index]]
            for index in indexs]
    svmsvc = sklearn.svm.SVC(kernel='precomputed')
    svmsvc.fit(kernel, labels)
    def tracks_filter(track):
        index = tracks_indexs.get(track)
        if index is None: return False
        kernel = metric[:, [index]].T.copy()
        labels = svmsvc.predict(kernel)
        return labels[0] in target_labels
    return tracks_filter

def source_tracks_reduce():
    def source_reduce(source_tracks):
        source_counts = collections.Counter()
        for source, tracks in source_tracks:
            source_counts[source] = len(tracks)
        return source_counts
    return source_reduce

def counts_tracks_reduce(tracks_weight=None):
    def counts_reduce(source_tracks):
        tracks_counts = collections.Counter()
        for source, tracks in source_tracks:
            tracks_counts.update(iter(tracks))
        if tracks_weight:
             tracks_counts = \
                 collections.Counter({
                     track: count *
                         tracks_weight
                             .get(track, 0)
                     for track, count in
                         tracks_counts.items()})
        return tracks_counts
    return counts_reduce

def common_tracks_reduce(
    tracks_filter,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_common=None,
    tracks_number=None,
    tracks_reduce=None,
    artist_reduce=None,
    artist_expand=None,
    artist_weight=None,
    artist_cutoff=None,
    return_sorted=None,
    return_source=None,
    return_fscore=None,
    return_common=None,
    return_number=None,
    return_tracks=None
):
    assert not callable(tracks_filter)
    _tracks_filter = set()
    if not artist_expand or artist_weight:
        _tracks_filter |= \
            expand_artist(set(tracks_filter))
    if artist_expand:
        _tracks_filter |= {
            artist for track in tracks_filter
                for artist in track_artist(track)}
    tracks_filter = _tracks_filter
    if tracks_weight:
        tracks_weight = \
            collections.defaultdict(
                functools.partial(int, 1),
                tracks_weight)
    if callable(tracks_reduce):
        return_tracks = True
    if not artist_expand:
        if artist_reduce is None:
            artist_reduce = True
    else:
        if not artist_weight:
            artist_reduce = None
    if return_sorted != False and \
        not return_fscore and \
        not return_common and \
        not return_number:
        return_sorted = True
        if return_fscore != False:
            return_fscore = True
        if return_common != False:
            return_common = True
        if return_number != False:
            return_number = True
    def common_reduce(source_tracks):
        result = []
        class _dict(dict):
            def __lt__(self, other):
                return len(self) < len(other)
        for source, tracks in source_tracks:
            tracks_source = set()
            if not artist_expand or \
                artist_weight:
                tracks_source |= \
                    tracks.keys() | \
                        expand_artist(
                            tracks.keys() -
                                tracks_filter,
                            collab=False)
                common0 = \
                    len(tracks_source
                        & tracks_filter)
            if artist_expand:
                tracks_source |= {
                    artist for track in tracks
                    if not artist_weight or
                        track not in tracks_filter
                        for artist in
                            track_artist(track)}
            common_tracks = \
                tracks_source & tracks_filter
            if not common_tracks: continue
            if artist_reduce:
                common_tracks = {
                    track_artist(track)[0]
                    for track in common_tracks}
            common1 = len(common_tracks)
            if not artist_expand or \
                not artist_weight:
                if not tracks_weight:
                    common = len(common_tracks)
                else:
                    common = sum(map(
                        tracks_weight.__getitem__,
                        common_tracks))
            else:
                counts = collections.Counter()
                for track in common_tracks:
                    artist = \
                        track_artist(track)[0]
                    counts[artist] += (
                        artist_weight
                        if ' - ' not in track else
                        tracks_weight[track]
                        if tracks_weight else 1)
                    counts[artist] = \
                        min(counts[artist],
                            artist_cutoff or 1)
                common = sum(counts.values())
            if tracks_fscore or return_fscore:
                precis = common/len(tracks)
                recall = common/len(tracks_filter)
                _fscore = fscore(
                    precis, recall, tracks_recall)
                if tracks_fscore:
                    if _fscore < tracks_fscore:
                        continue
            if tracks_common:
                if common < tracks_common:
                    continue
            if tracks_number:
                if len(tracks_source) \
                    < tracks_number: continue
            scores = []
            if return_fscore:
                scores.append(round(_fscore, 3))
            if return_common:
                if not artist_weight:
                    scores.append(common)
                else:
                    scores.append((
                        common, common0, common1))
            if return_number:
                scores.append(len(tracks))
            if return_tracks:
                if return_sorted:
                    tracks = _dict(tracks)
                scores.append(tracks)
            if len(scores) > 1:
                scores = (tuple(scores),)
            result.append(
                (source, *scores)
                    if scores else source)
        if isinstance(return_source, int):
            if return_sorted:
                if return_fscore or \
                    return_common or \
                    return_number:
                    result = \
                        collections.Counter(
                            dict(result)
                            ).most_common(
                                return_source)
            result = result[:return_source]
        if not callable(tracks_reduce):
            if return_fscore or \
                return_common or \
                return_number:
                result = \
                    collections.Counter(
                        dict(result))
        else:
            result = (
                (source, scores[-1]
                    if type(scores) is tuple
                        else scores
                ) for source, scores in result)
            result = tracks_reduce(result)
        return result
    return common_reduce

def impact_tracks_reduce(tracks_filter):
    if not isinstance(tracks_filter, set):
        tracks_filter = set(tracks_filter)
    def impact_reduce(source_tracks):
        tracks_impact = collections.Counter()
        for source, tracks in source_tracks:
            tracks_common = \
                tracks.keys() & tracks_filter
            tracks_weight = \
                collections.Counter(
                    dict.fromkeys(
                        tracks_common,
                        1 / len(tracks_common)))
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
    tracks_window=1,
    tracks_weight=None,
    tracks_artist=None,
    tracks_proxim=None
):
    if not isinstance(tracks_filter, set):
        tracks_filter = set(tracks_filter)
    if not tracks_weight:
        tracks_weight = \
            dict.fromkeys(tracks_filter, 1)
    if tracks_artist:
        tracks_artist = {
            track: track_artist(track)[0]
                for track in tracks_filter}
        artist_counts = \
            collections.Counter(
                tracks_artist.values())
    def proxim_reduce(source_tracks):
        tracks_counts = collections.Counter()
        for source, tracks in source_tracks:
            common_tracks = \
                tracks.keys() & tracks_filter
            if not common_tracks: continue
            indexs = {}
            for track in common_tracks:
                weight = \
                    tracks_weight.get(track, 1) \
                        / len(tracks[track])
                if tracks_artist:
                    weight /= \
                        artist_counts[
                            tracks_artist[track]]
                for index in tracks[track]:
                    for shift in range(
                        -tracks_window,
                        tracks_window + 1):
                        if tracks_proxim:
                            proxim = 1 - (
                                abs(shift or 1)-1
                                ) / tracks_window
                        else:
                            proxim = 1
                        indexs[index + shift] = \
                            max(proxim * weight,
                                indexs.setdefault(
                                    index + shift,
                                    0))
            for track in tracks:
                weight = 0
                for _index in tracks[track]:
                    _weight = indexs.get(_index)
                    if _weight:
                        weight = \
                            max(_weight, weight)
                if weight:
                    tracks_counts[track] += weight
        for track, count in tracks_counts.items():
            tracks_counts[track] = round(count, 3)
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
                        string), False))
    if isinstance(tracks, dict):
        return type(tracks)(
            dict(
                filter_tracks(
                    list(tracks.items()),
                    string,
                    action,
                    negate)))
    else:
        return type(tracks)(
            filter(filter_entity, tracks))

def fscore(precis, recall, weight=1):
    if precis == 0 or recall == 0: return 0
    if weight == 0:
        weight_precis, weight_recall = 1, 0
    elif weight == math.inf:
        weight_precis, weight_recall = 0, 1
    else:
        weight_precis, weight_recall = \
            1 / (weight or 0.5) ** 2, 1
    return \
        (weight_precis + weight_recall) \
            / (weight_precis / precis +
                weight_recall / recall)

def idf(counts, slicer=None, weight=None):
    if weight is None:
         maxima = counts.most_common(1)[0][1]
         def weight(count):
             if count == 0: return 0
             return 1 + math.log(
                 maxima / count, 10)
    if weight is False:
         def weight(count): return 1
    return type(counts)({
        track: weight(count)
            for track, count in (
                counts.items()
                    if slicer is None else
                counts.most_common()[slicer])})

def ndcg(ranked, seeded):
    ndcg, idcg, hits = 0, 0, 0
    for index, id in enumerate(ranked):
        if isinstance(id, tuple): id = id[0]
        if id in seeded:
            ndcg += 1 / math.log(index+2, 2)
            hits += 1
        idcg += 1 / math.log(index+2, 2)
    return (
        round(ndcg / idcg, 3),
        round(hits / len(ranked), 3))




def user2item(
    source_prefix=None,
    source_filter=None,
    source_number=None,
    source_debias=None,
    tracks_filter=None,
    tracks_window=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_recall=None,
    tracks_common=None,
    tracks_artist=None,
    tracks_proxim=None,
    tracks_debias=None,
    tracks_mapper=None,
    thread_number=None
):
    if source_number:
        if not tracks_fscore and \
            not tracks_common:
            tracks_fscore = True
    if tracks_filter:
        if tracks_window is None:
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
        thread_number=thread_number)
    if source_debias or tracks_debias:
        tracks_counts = \
            return_source(
                tracks_window=tracks_window
                    if tracks_debias else 0,
                tracks_reduce=
                    counts_tracks_reduce(),
                **args)
    if source_debias:
        args |= \
            dict(tracks_weight=idf(tracks_counts))
    if tracks_debias:
        tracks_weight = {
            track: 1 / count
                for track, count in
                    tracks_counts.items()}
    if tracks_proxim or tracks_debias:
        tracks_reduce = \
            proxim_tracks_reduce(
                tracks_filter,
                tracks_weight,
                tracks_artist,
                tracks_window,
                tracks_proxim)
    else:
        tracks_reduce = counts_tracks_reduce()
    return return_source(
        tracks_window=tracks_window,
        tracks_reduce=tracks_reduce,
        **args)




def item2item(
    caches_suffix,
    cached_metric=None,
    update_metric=None,
    source_prefix=None,
    source_filter=None,
    source_number=None,
    tracks_filter=None,
    tracks_window=None,
    tracks_weight=None,
    tracks_fscore=None,
    tracks_common=None,
    tracks_mapper=None,
    tracks_reduce=None,
    tracks_number=None,
    tracks_offset=None,
    lookup_transp=None,
    thread_number=None
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
            tracks_window=tracks_window,
            tracks_weight=tracks_weight,
            tracks_fscore=tracks_fscore,
            tracks_common=tracks_common,
            tracks_mapper=tracks_mapper,
            tracks_reduce=_tracks_reduce,
            thread_number=1)
        if tracks_number:
            tracks_counts = \
                dict(
                    tracks_counts.most_common(
                        tracks_number))
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
                map(reversed,
                    enumerate(source_counts)))
        tracks_source = {}
        def _tracks_reduce(source_tracks):
            def _tracks_reduce(source_tracks):
                if tracks_reduce:
                    source_tracks = \
                        tracks_reduce(
                            source_tracks)
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
                            in tracks_counts),
            tracks_mapper=tracks_mapper,
            tracks_reduce=_tracks_reduce,
            thread_number=1)
        tracks_source = [
            tracks_source.get(track, set())
                for track in tracks_counts]
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
                dtype='float16')
        common_vector = \
            numpy.zeros(
                shape=(len(tracks_counts),),
                dtype='float16')
        track1, count1 = \
            next(
                itertools.islice(
                    tracks_counts.items(),
                    index1,
                    None))
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
                        source_weight, 1)
            source_ranked = \
                dict(
                    map(reversed,
                        enumerate(source_counts)))
            source_weight = {
                source_ranked.get(
                    source, source): weight
                    for source, weight in
                        source_weight.items()}
            def return_source_counts(source):
                counts = \
                    filter(
                        None, map(
                            source_weight.get,
                            source))
                if power0 != 1:
                    counts = (
                        count ** power0
                            for count in counts)
                return sum(counts)
        if source_weight and normed:
            count1 = \
                return_source_counts(
                    tracks_source[index1])
            if not count1:
                return metric_vector, \
                    common_vector
        if metric is not None:
            print(os.getpid(), index1, track1)
        for index2, (track2, count2) in \
            enumerate(tracks_counts.items()):
            if index1 == index2: continue
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
                        tracks_source[index2])
                if not count2: continue
            source_common = \
                tracks_source[index1] \
                    & tracks_source[index2]
            if not source_common: continue
            if not source_weight:
                count0 = len(source_common)
            else:
                count0 = \
                    return_source_counts(
                        source_common)
                if not count0: continue
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
                round(result, 3)
            _result = count0 \
                / count1 ** power1 \
                    / count2 ** power2
            if not invert: _result = 1 / _result
            common_vector[index2] = \
                round(_result, 3)
        return metric_vector, common_vector
    def worker(number):
        source_counts, tracks_counts = \
            return_counts()
        tracks_source = return_indexs()
        metric, common = metric_common(mode='r+')
        for index1 in range(
            tracks_offset or 0, len(tracks_counts)
        ):
            index = index1 % thread_number
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
                            common)
                metric.flush()
                common.flush()
    def obtain_tracks_metric():
        if thread_number == 1:
            result = map(worker, range(1))
        else:
            thread = \
                multiprocessing.pool.ThreadPool(
                    thread_number)
            result = \
                thread.imap_unordered(
                    worker, range(thread_number))
        try:
            _ = list(result)
            if thread_number > 1:
                thread.close()
                thread.join()
        except Exception, KeyboardInterrupt:
            if thread_number > 1:
                thread.terminate()
                thread.join()
            raise
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
                len(tracks_counts)),
            dtype='float16')
        common_cached = \
            f'common.{caches_suffix}.memmap'
        if not os.path.exists(common_cached):
            if mode == 'r': return metric, None
            else: os.mknod(common_cached)
        common = numpy.memmap(
            common_cached,
            mode=mode,
            shape=(
                len(tracks_counts),
                len(tracks_counts)),
            dtype='float16')
        return metric, common
    if not thread_number: thread_number = 1
    source_counts, tracks_counts = \
        return_counts(update=True)
    ranked_source = list(source_counts)
    source_ranked = \
        dict(
            map(reversed,
                enumerate(source_counts)))
    ranked_tracks = list(tracks_counts)
    tracks_ranked = \
        dict(
            map(reversed,
                enumerate(tracks_counts)))
    tracks_source = return_indexs(update=True)
    indexs_artist = {}
    source_indexs = {}
    if cached_metric:
        if update_metric: obtain_tracks_metric()
        _metric, _common = metric_common(mode='r')
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
            if index1 is None: return None
        else:
            index1 = [
                tracks_ranked[track]
                    for track in track1
                        if track in tracks_ranked]
        if _metric is not None:
            metric_vector = _metric[index1]
            if _common is not None:
                common_vector = _common[index1]
            else:
                common_vector = numpy.zeros(
                    shape=(len(tracks_counts),),
                    dtype='float16')
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
                    power2=power2)
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
                    shape=(2, len(tracks_counts)),
                    dtype=float)
            if not metric and not common:
                metric = True
            if metric:
                ranker(metric_vector, matrix[0])
                if not common:
                    ranked_vector = matrix[0]
            if common:
                ranker(common_vector, matrix[1])
                if not metric:
                    ranked_vector = matrix[1]
            if metric and common:
                ranked_vector = \
                    scipy.stats.hmean(matrix)
        else:
            ranked_vector = itertools.repeat(0)
        return collections.Counter(
            dict((track2,
                ((track2_ranked,)
                    if ranked else ()) +
                ((track2_metric, track2_common)
                    if metric or not common else
                (track2_common, track2_metric)))
                for track2,
                    track2_metric,
                    track2_common,
                    track2_ranked in zip(
                        tracks_counts,
                        metric_vector,
                        common_vector,
                        ranked_vector)
                if track2_metric and (
                    type(metric) != int or
                        track2_metric >= metric
                ) and (
                    type(common) != int or
                        track2_common >= common)))
    return tracks_metric




def list2item(
    tracks_seeded,
    tracks_nearby,
    tracks_picked,
    tracks_artist
):
    tracks_seeded = \
        expand_artist(tracks_seeded, collab=True)
    tracks_seeded -= ignore_tracks()
    if tracks_artist:
        def artist(track):
            if type(track) == tuple:
                track = track[0]
            return track_artist(track)[0]
        tracks_seeded |= \
            set(map(artist, tracks_seeded))
    def artist_reduce(counts):
        counts = \
            sorted(counts.items(), key=artist)
        def _sum(scores):
            return sum(
                map(
                    lambda _:
                    collections.Counter({
                        track: score[0]
                            for track, score
                                in _.items()
                        }), scores),
                collections.Counter())
        return dict(map(
            lambda _: (_[0],
                _sum(map(
                    operator.itemgetter(1),
                    _[1]))),
            itertools.groupby(counts, artist)))
    if tracks_artist:
        tracks_nearby = \
            artist_reduce(tracks_nearby)
    artist_cached = {}
    def artist_differ(track1, track2):
        artist_cached.setdefault(
            track1, set(track_artist(track1)))
        artist_cached.setdefault(
            track2, set(track_artist(track2)))
        return artist_cached[track1] \
            .isdisjoint(artist_cached[track2])
    return collections.Counter(
        itertools.chain(*map(
            lambda track1:
            itertools.islice(
                filter(
                    lambda track2:
                    track2 not in tracks_seeded
                        and artist_differ(
                            track1, track2),
                    dict(
                        tracks_nearby[track1].
                            most_common()
                    )), tracks_picked),
            filter(
                lambda track:
                not tracks_seeded.isdisjoint(
                    expand_artist(track)),
                tracks_nearby))))

def select_tracks(
    tracks_seeded,
    tracks_nearby,
    tracks_picked,
    tracks_thrown=None,
    tracks_common=3,
    tracks_artist=True
):
    def cutoff(counts):
        return [
            track for track, count in
                counts.most_common()
                if count >= tracks_common]
    tracks = \
        cutoff(
            list2item(
                tracks_seeded,
                tracks_nearby,
                tracks_picked,
                tracks_artist))
    if tracks_thrown:
        if tracks_thrown < 0:
            tracks_thrown += tracks_picked
        _tracks = \
            cutoff(
                list2item(
                    tracks_seeded,
                    tracks_nearby,
                    tracks_thrown,
                    tracks_artist))
        tracks = [
            track for track in tracks
                if track not in _tracks]
    return tracks

def nearby_tracks(
    tracks_seeded,
    tracks_metric,
    tracks_picked=250
):
    return {
        track: collections.Counter(
            dict(
                (tracks_metric(track) or
                    collections.Counter())
                    .most_common(tracks_picked)
            )) for track in tracks_seeded}

def impact_tracks(
    tracks_seeded,
    tracks_nearby,
    tracks_picked=250
):
    if isinstance(tracks_seeded, str):
        tracks_seeded = {tracks_seeded}
    return list(reversed(
        collections.Counter({
            track: index
            for track, counts in
                tracks_nearby.items()
                for index, count in enumerate(
                    counts.most_common(
                        tracks_picked)
                ) if count[0] in tracks_seeded
            }).most_common()))




def source_id(source):
    if isinstance(source, int):
        return source
    if isinstance(source, tuple):
        return source_id(source[0]), *source[1:]
    if isinstance(source, set):
        return set(map(source_id, source))
    if isinstance(source, dict):
        return type(source)(
            dict(zip(
                map(source_id, source),
                source.values())))
    if not isinstance(source, str):
        return list(map(source_id, source))
    _, source = os.path.split(source)
    if source.endswith('.json'):
        source = source.rsplit('.json', 1)[0]
    elif source.endswith('.pickle'):
        source = source.rsplit('.pickle', 1)[0]
    if source.isdigit():
        source = int(source)
    return source

def unique_source(source):
    def file_name(file):
        return os.path.split(file)[-1]
    indexs = {}
    for file in source:
        indexs[file_name(file)] = file
    if isinstance(source, dict):
        return type(source)(
            dict(zip(
                indexs.values(),
                map(source.get, indexs.values())
                )))
    else:
        return list(indexs.values())

def locate_source(source='*', prefix=''):
    name, source_filter = '*.json', None
    if source != '*':
        id = source_id(source)
        if isinstance(id, (int, str)):
            name = f'{id}.json'
        else:
            def source_filter(file):
                return source_id(file) in id
    if isinstance(prefix, str): prefix = [prefix]
    prefix = [
        path if path.endswith('.json') else
            os.path.join(path or '**', name)
                for path in prefix]
    return (
        file for path in prefix
            for file in (
                (path,) if '*' not in path else
                glob.iglob(
                    path, recursive='**' in path
                )) if source_filter is None
                    or source_filter(file))

def locate_pickle(
    source_prefix,
    tracks_filter=None,
    tracks_weight=None,
    tracks_artist=None,
    return_cmdarg=None
):
    if isinstance(source_prefix, str):
        source_prefix = [source_prefix]
    prefix_filter = [
        prefix.removesuffix('.json') + '.pickle'
            if prefix.endswith('.json')
        else os.path.join(prefix, '*.pickle')
            for prefix in source_prefix]
    prefix_target = {
        prefix.removesuffix('.json') + '.pickle'
            if prefix.endswith('.json') else
        os.path.split(prefix.split('*')[0])[0]
            if '*' in prefix else prefix
            for prefix in source_prefix}
    _prefix = ''
    for prefix in sorted(prefix_target):
        if prefix.startswith(_prefix + os.sep):
            prefix_target.remove(prefix)
        else: _prefix = prefix
    if not tracks_filter:
        _tracks_filter = ['(?-u:\\x94]\\x94)']
    else:
        _tracks_filter = [
            ('(?-u:\\x8c\\x{:02x})'
                .format(pickle.dumps(track)[12])
                if not ord('\n') <=
                    pickle.dumps(track)[12]
                        <= ord('\r') else '')
            + regex.escape(track, True, True)
            + '(?-u:\\x94)'
                for track in tracks_filter]
    args = (
        'rg',
        *(  ('--count-matches',)
            if not tracks_filter else
            ('--only-matching',
            '--no-line-number')),
        '--with-filename',
        '--no-heading',
        '--text',
        *itertools.chain(
            *zip(itertools.repeat('-e'),
                _tracks_filter)),
        *itertools.chain(
            *zip(itertools.repeat('-g'),
                prefix_filter)),
        *prefix_target)
    if return_cmdarg: return args
    try:
        proc = \
            subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                text=True,
                errors='ignore')
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
                weight = (
                    1 if not tracks_weight
                        else tracks_weight.get(
                            string, 1))
                if not tracks_artist:
                    source_counts[source] += \
                        weight
                else:
                    artist = \
                        track_artist(string)[0]
                    counts = \
                        source_counts.setdefault(
                            source, dict())
                    counts[artist] = \
                        max(weight,
                            counts.get(artist, 0))
        if tracks_artist:
            for source, counts in \
                source_counts.items():
                source_counts[source] \
                    = sum(counts.values())
        return source_counts
    except OSError:
        traceback.print_exc()
        return None




def return_tracks(
    source,
    prefix='',
    sorted=True,
    pickle=True,
    **kwargs
):
    source = locate_source(source, prefix)
    master = next(source, None)
    if master is None: return None
    if sorted and os.sep in master:
        master = \
            max((master, *source),
                key=os.path.getsize)
    if pickle:
        if not os.path.getsize(master):
            return import_source_pickle(
                master, **kwargs)
    return import_source(master, **kwargs)

def import_source(
    source,
    filter=None,
    window=None,
    number=None,
    mapper=None,
    pickle=None
):
    def mapped(tracks):
        if not mapper: return tracks
        def sorter(_): return mapper(_[0])
        return {
            track: sorted(
                itertools.chain(
                    *map(operator.itemgetter(1),
                        group)))
            for track, group in
                itertools.groupby(
                    sorted(
                        tracks.items(), key=sorter
                    ), key=sorter)}
    if pickle:
        tracks = \
            import_source_pickle(
                source,
                filter=filter,
                window=window,
                number=number)
        if tracks is not None:
            return mapped(tracks)
    with open(source) as f:
        tracks = \
            import_tracks(
                itertools.islice(
                    ijson.items(f, 'item'),
                    number),
                filter=filter,
                window=window)
    if pickle:
        if (filter is None or
            (window == math.inf and tracks)) \
            and (number is None or
                len(tracks) <= number):
            export_source_pickle(tracks, source)
    if mapper: tracks = mapped(tracks)
    return tracks

def import_tracks(
    tracks,
    filter=None,
    window=None
):
    _filter = filter
    if callable(filter):
        filter = lambda _: _filter(_[1])
    elif filter:
        filter = lambda _: _[1] in _filter
    return dict(
        functools.reduce(
            lambda _tracks, track:
                _tracks[track[1]]
                    .append(track[0]) or _tracks,
            tracks_picker(
                enumerate(
                    map(lambda track:
                        track_id(**track),
                        tracks)),
                filter,
                window),
            collections.defaultdict(list)))

def tracks_picker(
    tracks,
    filter,
    window
):
    if window:
        maxlen = window
        if window == math.inf: maxlen = None
        buffer = collections.deque(maxlen=maxlen)
    tracks = iter(tracks)
    for track in tracks:
        if not filter or filter(track):
            if window:
                yield from buffer
                buffer.clear()
            yield track
            if window:
                yield from itertools.islice(
                    tracks, maxlen)
        else:
            if window: buffer.append(track)

def export_source_pickle(
    tracks,
    source,
    sorted=True
):
    source = \
        source.removesuffix('.json') + '.pickle'
    with open(source, 'wb') as f:
        pickle.dump(
            tracks if not sorted else
            dict(__builtins__.sorted(
                tracks.items())), f)

def import_source_pickle(
    source,
    filter=None,
    window=None,
    number=None
):
    source_pickle = \
        source.removesuffix('.json') + '.pickle'
    if not os.path.exists(source_pickle):
        return None
    if os.path.getsize(source):
        if os.path.getmtime(source) > \
            os.path.getmtime(source_pickle):
            return None
    with open(source_pickle, 'rb') as f:
        tracks = pickle.load(f)
        if filter is None and number is None:
            return tracks
        idxs = set()
        for track in (
            tracks if callable(filter) else
            filter if filter else tracks
        ):
            if (filter(track)
                if callable(filter) else
                track in tracks
                if filter is not None else True
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
                                idx + window + 1))
        return {
            track: index
                for track, index in tracks.items()
                    if (any(
                        builtins.filter(
                            idxs.__contains__,
                            index))
                        if window != math.inf
                            else idxs
                    ) and (
                        number is None or
                            min(index) < number)}




def track_id(
    *,
    artist,
    title,
    subtitle=None,
    **kwargs
):
    if type(artist) is list:
        artist = ', '.join(artist)
    artist = adjust_artist(artist)
    title = adjust_title(title)
    id = f'{artist} - {title}'
    if subtitle:
        subtitle = adjust_string(subtitle)
        if ((
            any(
                map(subtitle.__contains__,
                    tracks_tweaks)
            ) and not any(
                map(subtitle.__contains__,
                    tracks_traits)
            )) or (any(
                map(subtitle.__contains__,
                    tracks_medias)
            ) and any(
                map(title.__contains__,
                    tracks_themes)
            ))): return id + f' ({subtitle})'
    return id

def track_artist(id):
    artist, *_ = id.split(' - ', 1)
    for lexema, titles, phrase, spaces \
        in tracks_collab:
        for _spaces, _lexema in \
            ((1, f' {lexema} '), (0, lexema)):
            if spaces is not None:
                if _spaces != spaces: continue
            if _lexema not in artist: continue
            string1, string2 = \
                artist.split(_lexema, 1)
            string1 = string1.strip()
            artist1 = track_artist(string1)
            artist2 = track_artist(string2)
            if phrase and ' ' not in string1:
                artist1 = []
                artist2[0] = \
                    string1 + _lexema + artist2[0]
            return artist1 + artist2
    return [artist.strip()]

def track_title(id):
    return id.split(' - ', 1)[1]

def adjust_artist(artist):
    artist = adjust_string(artist)
    artist = artist.split(' - ')[-1]
    if '(' in artist and ')' in artist:
        artist, _ = artist.split('(', 1)
        artist += _.split(')', 1)[-1]
        artist = adjust_artist(artist)
    return artist.strip()

def adjust_title(title):
    title = adjust_string(title)
    naming, *parens = title.rsplit('(', 1)
    for lexema, titles, phrase, spaces \
        in tracks_collab:
        if not titles: continue
        naming = naming.split(f' {lexema} ')[0]
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
                    any(map(
                        naming.__contains__,
                        tracks_themes)): continue
                title = naming
                break
        if title == naming: ending = ''
        else: ending = f' ({parens}'
        title = adjust_title(naming) + ending
    return title.strip()

def adjust_string(string):
    string = ' '.join(string.split())
    string = \
        ''.join(
            filter(
                lambda _:
                    unicodedata.category
                        (_)[0] != 'C',
                string))
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
            string = string.removesuffix(suffix)
        if string == _string: break
    return string

def reduce_artist(tracks):
    if isinstance(tracks, str):
        artist = track_artist(tracks)
        if not artist[1:]: return tracks
        return track_id(
            artist=artist[0],
            title=track_title(tracks))
    return set(map(reduce_artist, tracks))

def expand_artist(tracks, collab=None):
    if isinstance(tracks, str):
        tracks = {tracks}
    else:
        tracks = set(tracks)
    for track in list(tracks):
        artist = track_artist(track)
        if not artist[1:]: continue
        title = track_title(track)
        for _artist in artist:
            tracks.add(
                track_id(
                    artist=_artist, title=title))
        if collab in (None, True):
            collab = \
                (', ', ' & ', ' and ', ' feat. ')
        for _collab in collab or []:
            tracks.add(
                track_id(
                    artist=
                        _collab.join(artist[:2]),
                    title=title))
    return tracks

tracks_collab = (
    ('featuring', True, None, True),
    ('feat.', True, None, True),
    ('feat', True, None, True),
    ('ft.', True, None, True),
    ('ft', True, None, True),
    ('prod.', True, None, True),
    ('prod', True, None, True),
    ('pres.', True, None, True),
    ('pres', True, None, True),
    ('and', False, True, True),
    ('with', False, None, True),
    ('w/', False, None, True),
    ('vs.', False, None, True),
    ('vs', False, None, True),
    ('x', False, None, True),
    ('&', False, True, True),
    (',', False, None, None),
    ('/', False, None, None),
    ('\\', False, None, None)
)

tracks_tweaks = (
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
    'version',
    're-version',
    'acoustic',
    'single',
    'album',
    'radio',
    'club',
    'edit',
    'excerpt',
    'extract',
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

def ignore_artist():
    return {'разные исполнители'}

def ignore_tracks():
    return {
        'unknown artist - untitled',
        'неизвестен - без названия'
    }