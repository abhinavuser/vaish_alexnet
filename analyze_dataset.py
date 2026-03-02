
"""
Comprehensive analysis of the Tomato Leaves Dataset
"""

import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_dataset():


    base_path = Path('Tomato_Leaves')
    images_path = base_path / 'Images'
    labels_path = base_path / 'labels'

    results = {
        'summary': {},
        'training': {},
        'validation': {},
        'labels_analysis': {},
        'data_quality': {}
    }


    train_img_path = images_path / 'train'
    train_label_path = labels_path / 'train'

    train_images = sorted([f.name for f in train_img_path.iterdir() if f.is_file()])
    train_labels = sorted([f.name for f in train_label_path.iterdir() if f.is_file()])

    results['training']['total_images'] = len(train_images)
    results['training']['total_labels'] = len(train_labels)


    val_img_path = images_path / 'val'
    val_label_path = labels_path / 'val'

    val_images = sorted([f.name for f in val_img_path.iterdir() if f.is_file()])
    val_labels = sorted([f.name for f in val_label_path.iterdir() if f.is_file()])

    results['validation']['total_images'] = len(val_images)
    results['validation']['total_labels'] = len(val_labels)


    total_images = len(train_images) + len(val_images)
    total_labels = len(train_labels) + len(val_labels)

    results['summary']['total_images'] = total_images
    results['summary']['total_labels'] = total_labels


    def count_labels(label_path, label_dir_name):

        healthy_count = 0
        diseased_count = 0

        for label_file in (label_path / label_dir_name).iterdir():
            if label_file.is_file() and label_file.suffix == '.txt':
                filename = label_file.name

                if filename.startswith('H'):
                    healthy_count += 1
                elif filename.startswith('D'):
                    diseased_count += 1

        return healthy_count, diseased_count


    train_healthy, train_diseased = count_labels(labels_path, 'train')
    results['labels_analysis']['training'] = {
        'healthy': train_healthy,
        'diseased': train_diseased,
        'total_labeled': train_healthy + train_diseased
    }


    val_healthy, val_diseased = count_labels(labels_path, 'val')
    results['labels_analysis']['validation'] = {
        'healthy': val_healthy,
        'diseased': val_diseased,
        'total_labeled': val_healthy + val_diseased
    }


    total_healthy = train_healthy + val_healthy
    total_diseased = train_diseased + val_diseased

    results['labels_analysis']['overall'] = {
        'healthy': total_healthy,
        'diseased': total_diseased,
        'total': total_healthy + total_diseased
    }


    missing_labels_train = []
    for img in train_images:
        label_name = img.rsplit('.', 1)[0] + '.txt'
        if label_name not in [l.name for l in train_label_path.iterdir()]:
            missing_labels_train.append(img)

    missing_labels_val = []
    for img in val_images:
        label_name = img.rsplit('.', 1)[0] + '.txt'
        if label_name not in [l.name for l in val_label_path.iterdir()]:
            missing_labels_val.append(img)

    results['data_quality']['missing_labels_training'] = len(missing_labels_train)
    results['data_quality']['missing_labels_validation'] = len(missing_labels_val)
    results['data_quality']['missing_labels_training_files'] = missing_labels_train[:10]
    results['data_quality']['missing_labels_validation_files'] = missing_labels_val[:10]

    return results

def print_results(results):


    print("=" * 70)
    print("Tomato leaves dataset analysis report")
    print("=" * 70)

    print("\n1. DATASET SUMMARY")
    print("-" * 70)
    print(f"   Total Images:         {results['summary']['total_images']:,}")
    print(f"   Total Label Files:    {results['summary']['total_labels']:,}")

    print("\n2. TRAINING SET STATISTICS")
    print("-" * 70)
    print(f"   Total Images:         {results['training']['total_images']:,}")
    print(f"   Total Label Files:    {results['training']['total_labels']:,}")
    train_total = results['labels_analysis']['training']['total_labeled']
    print(f"   Healthy Leaves:       {results['labels_analysis']['training']['healthy']:,}")
    print(f"   Diseased Leaves:      {results['labels_analysis']['training']['diseased']:,}")
    if train_total > 0:
        print(f"   Percentage Healthy:   {results['labels_analysis']['training']['healthy']/train_total*100:.2f}%")
        print(f"   Percentage Diseased:  {results['labels_analysis']['training']['diseased']/train_total*100:.2f}%")

    print("\n3. VALIDATION SET STATISTICS")
    print("-" * 70)
    print(f"   Total Images:         {results['validation']['total_images']:,}")
    print(f"   Total Label Files:    {results['validation']['total_labels']:,}")
    val_total = results['labels_analysis']['validation']['total_labeled']
    print(f"   Healthy Leaves:       {results['labels_analysis']['validation']['healthy']:,}")
    print(f"   Diseased Leaves:      {results['labels_analysis']['validation']['diseased']:,}")
    if val_total > 0:
        print(f"   Percentage Healthy:   {results['labels_analysis']['validation']['healthy']/val_total*100:.2f}%")
        print(f"   Percentage Diseased:  {results['labels_analysis']['validation']['diseased']/val_total*100:.2f}%")

    print("\n4. OVERALL LABEL DISTRIBUTION")
    print("-" * 70)
    overall = results['labels_analysis']['overall']
    total = overall['total']
    print(f"   Total Healthy Images: {overall['healthy']:,} ({overall['healthy']/total*100:.2f}%)")
    print(f"   Total Diseased Images: {overall['diseased']:,} ({overall['diseased']/total*100:.2f}%)")

    print("\n5. DATA QUALITY ASSESSMENT")
    print("-" * 70)
    print(f"   Missing Labels (Training): {results['data_quality']['missing_labels_training']}")
    print(f"   Missing Labels (Validation): {results['data_quality']['missing_labels_validation']}")

    if results['data_quality']['missing_labels_training'] > 0:
        print(f"   Sample Missing (Training): {results['data_quality']['missing_labels_training_files'][:3]}")

    if results['data_quality']['missing_labels_validation'] > 0:
        print(f"   Sample Missing (Validation): {results['data_quality']['missing_labels_validation_files'][:3]}")

    print("\n6. DATASET BALANCE RATIO")
    print("-" * 70)
    if results['labels_analysis']['overall']['healthy'] > 0:
        ratio = results['labels_analysis']['overall']['diseased'] / results['labels_analysis']['overall']['healthy']
        print(f"   Diseased to Healthy Ratio: {ratio:.2f}:1")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    results = analyze_dataset()
    print_results(results)


    with open('dataset_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Analysis saved to 'dataset_analysis.json'")
