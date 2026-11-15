"""Offline tests for efficacy rubrics (no NDIF)."""
from eval_bank import BANK, Rubric, items_by_id
from eval_scoring import connected_helps, score_item, score_reply


def test_bank_ids_unique():
    ids = [item.id for item in BANK]
    assert len(ids) == len(set(ids))


def test_memory_item_scores_gold():
    item = items_by_id(["mem_room_preference"])[0]
    sc = score_item(item.gold, item)
    assert sc.pass_, sc.detail


def test_forbidden_catches_room4():
    item = items_by_id(["mem_room_preference"])[0]
    sc = score_item("Just book Room 4 again.", item)
    assert not sc.pass_
    assert sc.hit_forbidden >= 1


def test_connected_helps_when_solo_fails():
    item = items_by_id(["env_status_saved"])[0]
    solo = score_item("I have no idea.", item)
    conn = score_item("It Saved.", item)
    assert connected_helps(solo, conn, item) is True


def test_plain_chat_does_not_require_peers():
    item = items_by_id(["plain_chat_capital"])[0]
    assert item.rubric.needs_peers is False
    assert score_item("Paris", item).pass_


def test_empty_rubric_pass_on_text():
    sc = score_reply("ok", Rubric())
    assert sc.pass_


def test_all_golds_pass():
    for item in BANK:
        sc = score_item(item.gold, item)
        assert sc.pass_, f"{item.id}: {sc.detail} gold={item.gold!r}"
