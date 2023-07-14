from dataset import DPDataset
from pathlib import Path
import numpy as np
import random
import bisect


def subset_time(
    ds: DPDataset,
    train_ratio: float,
    val_test_ratio: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:

    val_ratio = (1 - train_ratio) * val_test_ratio

    classifier = lambda x: (Path(x.venue_path).stem, x.session, x.camera_id)
    data_info = [(classifier(d), i) for i, d in enumerate(ds.datalist)]
    video_list = sorted(list(set(d[0] for d in data_info)))
    video_idx_list = [
        list(map(lambda x: x[1], filter(lambda d: d[0] == c, data_info)))
        for c in video_list
    ]

    max_frames = dict()

    # データセット中の各セッションについて，最大フレームはいくつかを計算
    for video_info, video_idx in zip(video_list, video_idx_list):
        # video_info = (venue_path.stem, session, camera_id) なので，その2つ目まで
        venue_session_info = video_info[:2]
        # この動画内の最後のindexのframesの最初
        if venue_session_info not in max_frames:
            max_frames[venue_session_info] = ds.datalist[video_idx[-1]].frames[0]
        else:
            max_frames[venue_session_info] = max(
                ds.datalist[video_idx[-1]].frames[-1], max_frames[venue_session_info]
            )

    train_idx, val_idx, test_idx = [], [], []

    # 最大フレームを70:15:15で割るように結果を出力
    for video_info, video_idx in zip(video_list, video_idx_list):
        venue_session_info = video_info[:2]
        l = max_frames[venue_session_info]
        train_val_sep_frm = int(l * train_ratio)
        val_test_sep_frm = int(l * (train_ratio + val_ratio))
        frames_of_video = [ds.datalist[i].frames[0] for i in video_idx]

        train_val_sep = bisect.bisect_left(frames_of_video, train_val_sep_frm)
        val_test_sep = bisect.bisect_left(frames_of_video, val_test_sep_frm)

        train_idx.extend(video_idx[:train_val_sep])
        val_idx.extend(video_idx[train_val_sep:val_test_sep])
        test_idx.extend(video_idx[val_test_sep:])

    return train_idx, val_idx, test_idx


def subset_weight(train_ds):
    """
    Given two random_split datasets, returns weights of them on the occurrences of each class.
    """
    action_ids = np.array(list(map(lambda x: x.action_id, train_ds.dataset.datalist)))
    id_sample_count = np.array(
        [len(np.where(action_ids == i)[0]) for i in np.unique(action_ids)]
    )
    action_classes_sample_count = np.array(
        [
            np.isin(action_ids, v).sum()
            for v in train_ds.dataset.action_class_dict.values()
        ]
    )
    print(f"{id_sample_count=}")
    print(f"{action_classes_sample_count=}")
    class_weight = 1.0 / action_classes_sample_count
    train_weight = np.array(
        [
            class_weight[train_ds.dataset.rev_action_class_dict[action_ids[idx]]]
            for idx in train_ds.indices
        ]
    )
    train_weight /= train_weight.sum()
    return train_weight


def subset_venue(
    ds: DPDataset, training: str, val_test_ratio=0.5
) -> tuple[list[int], list[int], list[int]]:
    """
    training: If this value is "living", will train on livingroom and evaluate on openoffice. Otherwise, will train on openoffice and evaluate on livingroom.
    """
    classifier = lambda x: x.venue_path.split("-")[-1]
    data_info = [
        (classifier(d), i) for i, d in enumerate(ds.datalist)
    ]  # List of ('livingroom' or 'openoffice', index)
    train_idx = list(
        map(lambda x: x[1], filter(lambda d: d[0] == "livingroom", data_info))
    )
    val_idx = list(
        map(lambda x: x[1], filter(lambda d: d[0] == "openoffice", data_info))
    )
    if training != "living":
        train_idx, val_idx = val_idx, train_idx

    random.seed(42)
    random.shuffle(val_idx)

    l = len(val_idx)
    val_idx, test_idx = (
        val_idx[: int(l * val_test_ratio)],
        val_idx[int(l * val_test_ratio) :],
    )
    return train_idx, val_idx, test_idx


VENUESESSION2PERSONID = {
    # The 1st day has 6 people: 1-6
    ("2023-01-17-livingroom", "take1"): 1,
    ("2023-01-17-livingroom", "take2"): 2,
    ("2023-01-17-livingroom", "take3"): 3,
    ("2023-01-17-livingroom", "take4"): 4,
    ("2023-01-17-livingroom", "take5"): 5,
    ("2023-01-17-livingroom", "take6"): 6,
    ("2023-01-17-openoffice", "take1"): 3,
    ("2023-01-17-openoffice", "take2"): 1,
    ("2023-01-17-openoffice", "take4"): 6,
    ("2023-01-17-openoffice", "take5"): 5,
    ("2023-01-17-openoffice", "take6"): 4,
    # 2nd day has 6 people: 7-12
    ("2023-01-18-livingroom", "take1"): 7,
    ("2023-01-18-livingroom", "take2"): 8,
    ("2023-01-18-livingroom", "take3"): 9,
    ("2023-01-18-livingroom", "take4"): 10,
    ("2023-01-18-livingroom", "take5"): 11,
    ("2023-01-18-livingroom", "take6"): 12,
    ("2023-01-18-openoffice", "take4"): 12,
    ("2023-01-18-openoffice", "take5"): 10,
    ("2023-01-18-openoffice", "take6"): 11,
    # 3rd day has 6 ppl: 13-18
    ("2023-01-19-livingroom", "take1"): 13,
    ("2023-01-19-livingroom", "take2"): 14,
    ("2023-01-19-livingroom", "take3"): 15,
    ("2023-01-19-livingroom", "take4"): 16,
    ("2023-01-19-livingroom", "take5"): 17,
    ("2023-01-19-livingroom", "take6"): 18,
    ("2023-01-19-openoffice", "take4"): 17,
    # 01-24 has 6 people: 19-24
    ("2023-01-24-livingroom", "take1"): 20,
    ("2023-01-24-livingroom", "take2"): 21,
    ("2023-01-24-livingroom", "take4"): 23,
    ("2023-01-24-openoffice", "take1"): 19,
    ("2023-01-24-openoffice", "take2"): 20,
    ("2023-01-24-openoffice", "take3"): 21,
    ("2023-01-24-openoffice", "take4"): 22,
    ("2023-01-24-openoffice", "take5"): 23,
    ("2023-01-24-openoffice", "take6"): 24,
    # 01-25 has 9 people: 25-33
    ("2023-01-25-livingroom", "take1"): 25,
    ("2023-01-25-livingroom", "take2"): 26,
    ("2023-01-25-livingroom", "take3"): 27,
    ("2023-01-25-livingroom", "take4"): 28,
    ("2023-01-25-livingroom", "take6"): 33,
    ("2023-01-25-livingroom", "take9"): 29,
    ("2023-01-25-livingroom", "take10"): 30,
    ("2023-01-25-openoffice", "take1"): 27,
    ("2023-01-25-openoffice", "take6"): 29,
    ("2023-01-25-openoffice", "take7"): 31,
    ("2023-01-25-openoffice", "take8"): 32,
}


def subset_person(
    ds: DPDataset,
    val_people=4,
    tst_people=4,
) -> tuple[list[int], list[int], list[int]]:
    """同じ人はtrain/val/testのいずれかにしか出ないようにする"""
    classifier = lambda x: VENUESESSION2PERSONID[
        (
            Path(x.venue_path).stem,
            x.session,
        )  # ('date-livingroom' or 'date-openoffice', session name)
    ]
    data_info = [(classifier(d), i) for i, d in enumerate(ds.datalist)]
    unique_class = sorted(list(set(d[0] for d in data_info)))
    assert len(unique_class) == len(set(VENUESESSION2PERSONID.values()))

    random.seed(42)
    random.shuffle(unique_class)

    trn_people = len(unique_class) - val_people - tst_people
    trn_cls, val_cls, tst_cls = (
        unique_class[:trn_people],
        unique_class[trn_people : trn_people + val_people],
        unique_class[trn_people + val_people :],
    )

    trn_idx = list(map(lambda x: x[1], filter(lambda d: d[0] in trn_cls, data_info)))
    val_idx = list(map(lambda x: x[1], filter(lambda d: d[0] in val_cls, data_info)))
    tst_idx = list(map(lambda x: x[1], filter(lambda d: d[0] in tst_cls, data_info)))

    return trn_idx, val_idx, tst_idx


def val2ind(x, a, b, bins):
    """If range [a,b] is separated into `bins`, which bins `x` will fall into?"""
    return int((x - a) / (b - a) * bins)
