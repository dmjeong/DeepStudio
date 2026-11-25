"""Explicit repair of legacy projects that accidentally named background as a defect."""
from core.annotation_journal import guarded_dataset
from core.project import SEGMENTATION_BACKGROUND_NAMES


def needs_first_class_repair(project):
    names = project.data.class_names
    return (project.task == 'segment' and bool(names)
            and names[0].casefold() not in SEGMENTATION_BACKGROUND_NAMES)


@guarded_dataset
def repair_first_class(project):
    """Append the misplaced foreground; preserve every existing ID >= 1.

    Pixel zero by itself contains no recoverable object boundary. Only explicit
    paint shapes / polygon labels can recover the formerly invisible object.
    Files and project metadata are journaled together and rollback on failure.
    """
    from core.class_management import _walk_files, IMAGE_EXTENSIONS
    from core.dataset_editor import EditTransaction, SPLITS, split_root, sidecars, read_annotations
    from core.mask_annotations import load_mask_document

    if not needs_first_class_repair(project):
        return {'changed': False, 'restored_regions': 0}
    names = list(project.data.class_names)
    if len(names) >= 255:
        raise ValueError('클래스가 255개입니다. 사용하지 않는 전경 클래스를 삭제한 뒤 복구하세요.')
    reserved = next((name for name in ('background', '배경', 'bg')
                     if name.casefold() not in {n.casefold() for n in names}), None)
    if reserved is None:
        raise ValueError('예약 배경 이름(background, 배경, bg)이 모두 사용 중입니다. 이름을 정리하세요.')
    first_name, new_id = names[0], len(names)
    transaction = EditTransaction(project, 'restore-first-segmentation-class')
    restored = 0
    try:
        for split in SPLITS:
            for path in _walk_files(split_root(project, split)):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                related = sidecars(project, path, split)
                if not related:
                    continue
                document = load_mask_document(project, path, split)
                document.classes = len(names) + 1
                changed = 0
                for shape in document.shapes:
                    if shape['class_id'] == 0 and shape.get('operation') != 'erase':
                        shape['class_id'] = new_id
                        changed += 1
                    elif shape.get('operation') == 'erase' and shape.get('scope_class') == 0:
                        # Erasing the old class must still erase that same region
                        # after its paint ID moves, rather than restoring the hole.
                        shape['scope_class'] = new_id
                if changed:
                    document.invalidate()
                    mask_paths = [p for kind, p in related if kind == 'masks']
                    target = (mask_paths[0] if mask_paths else
                              split_root(project, split).parent.parent / 'masks' / split /
                              path.relative_to(split_root(project, split)).with_suffix('.png'))
                    transaction.write(target, document.encode())
                    restored += changed
                for kind, label in related:
                    if kind != 'labels':
                        continue
                    rows = read_annotations(label, 'segment', len(names))
                    if any(row['class_id'] == 0 for row in rows):
                        text = ''.join(' '.join([str(new_id if row['class_id'] == 0 else row['class_id']),
                                               *(format(v, '.12g') for v in row['coordinates'])]) + '\n'
                                       for row in rows)
                        transaction.write(label, text.encode('utf-8'))
        project.data.class_names = [reserved, *names[1:], first_name]
        project.data.num_classes = len(project.data.class_names)
        transaction.finish()
    except BaseException:
        transaction.rollback()
        raise
    return {'changed': True, 'class_name': first_name, 'class_id': new_id,
            'restored_regions': restored, 'backup': str(transaction.backup)}
