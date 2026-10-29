"""Turn st.data_editor's change record into database updates.

The editor state uses row POSITIONS; the shown table is indexed by database id,
so position p means id shown.index[p].
"""
import pandas as pd


def as_date_text(value):
    return pd.to_datetime(value, format="mixed").strftime("%Y-%m-%d 00:00:00")


def _same(col, old, new):
    if col == "date":
        return as_date_text(old) == as_date_text(new)
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        try:
            return round(float(old), 2) == round(float(new), 2)
        except (TypeError, ValueError):
            return False
    return str(old) == str(new)


def editor_changes(shown, state):
    ids = list(shown.index)
    state = state or {}
    deleted = [int(ids[int(p)]) for p in state.get("deleted_rows", [])]
    updates = {}
    for pos, changes in (state.get("edited_rows") or {}).items():
        row_id = int(ids[int(pos)])
        if row_id in deleted:
            continue      # edited and deleted only counts as deleted
        old = shown.loc[row_id].to_dict()
        if all(_same(c, old.get(c), v) for c, v in changes.items()):
            continue      # an "edit" that changes nothing
        updates[row_id] = {**old, **changes}
    return updates, deleted
