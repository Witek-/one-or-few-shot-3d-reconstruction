#!/usr/bin/env python3
"""Interactively measure distances between points from one or two images.

The script consumes ``metric_predictions.npz`` produced by
``reconstruct_cli_metric.py``. All views share one predicted world coordinate
frame, so endpoint A and endpoint B may be selected in different images.
No scale estimation or alignment is performed.
"""

import argparse
import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MEASUREMENT_SCHEMA_VERSION = "3"
LEGACY_MEASUREMENT_FIELDS_V2 = [
    "schema_version",
    "timestamp_utc",
    "prediction_archive",
    "scene_name",
    "subset_label",
    "view_count",
    "image_a_index",
    "image_a_name",
    "image_b_index",
    "image_b_name",
    "cross_view_measurement",
    "segment_name",
    "annotation_trial",
    "reference_distance_m",
    "reference_uncertainty_mm",
    "predicted_distance_m",
    "signed_error_m",
    "absolute_error_m",
    "absolute_error_cm",
    "percentage_error",
    "click_a_x",
    "click_a_y",
    "click_b_x",
    "click_b_y",
    "resolved_a_x",
    "resolved_a_y",
    "resolved_b_x",
    "resolved_b_y",
    "point_a_x_m",
    "point_a_y_m",
    "point_a_z_m",
    "point_b_x_m",
    "point_b_y_m",
    "point_b_z_m",
    "patch_radius_px",
    "valid_samples_a",
    "valid_samples_b",
    "confidence_a_median",
    "confidence_b_median",
    "metric_scaling_factor_a",
    "metric_scaling_factor_b",
    "sampling_method",
]
MEASUREMENT_FIELDS = [
    *LEGACY_MEASUREMENT_FIELDS_V2[:-1],
    "depth_z_a_m",
    "depth_z_b_m",
    LEGACY_MEASUREMENT_FIELDS_V2[-1],
]


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open a browser interface, select endpoints in one or two images, "
            "and append the metric distance and errors to CSV."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to metric_predictions.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV output path (default: metric_measurements_multiview.csv next "
            "to the NPZ)."
        ),
    )
    parser.add_argument(
        "--patch-radius",
        type=non_negative_int,
        default=2,
        help="Sampling radius in pixels; 2 means a 5x5 patch (default: 2).",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Gradio bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--server-port",
        type=positive_int,
        default=7860,
        help="Gradio port (default: 7860).",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a Gradio share URL."
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the interface in the default browser.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the archive and print its metadata without launching Gradio.",
    )
    return parser.parse_args()


def scalar_from_archive(archive, key, default):
    if key not in archive:
        return default
    value = np.asarray(archive[key])
    return value.item() if value.size == 1 else default


class MetricPredictionSet:
    REQUIRED_KEYS = {
        "world_points",
        "final_masks",
        "depth_z",
        "confidence",
        "metric_scaling_factor",
        "processed_images_uint8",
        "image_names",
        "metric_scale_already_applied",
    }

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Prediction archive does not exist: {self.path}")

        with np.load(self.path, allow_pickle=False) as archive:
            missing = sorted(self.REQUIRED_KEYS.difference(archive.files))
            if missing:
                raise ValueError(
                    "Archive is not a metric experiment export; missing arrays: "
                    + ", ".join(missing)
                )
            self.world_points = np.asarray(archive["world_points"], dtype=np.float32)
            self.final_masks = np.asarray(archive["final_masks"], dtype=bool)
            self.depth_z = np.asarray(archive["depth_z"], dtype=np.float32)
            self.confidence = np.asarray(archive["confidence"], dtype=np.float32)
            self.metric_scaling_factor = np.asarray(
                archive["metric_scaling_factor"],
                dtype=np.float32,
            )
            self.images = np.asarray(archive["processed_images_uint8"], dtype=np.uint8)
            self.image_names = [str(value) for value in archive["image_names"].tolist()]
            self.scene_name = str(
                scalar_from_archive(
                    archive, "experiment_scene_name", self.path.parent.name
                )
            )
            self.subset_label = str(
                scalar_from_archive(archive, "experiment_subset_label", "")
            )
            self.distance_units = str(
                scalar_from_archive(archive, "distance_units", "meters")
            )
            scale_applied = bool(
                scalar_from_archive(archive, "metric_scale_already_applied", False)
            )

        if not scale_applied:
            raise ValueError(
                "Archive does not confirm that metric scale is already applied. "
                "Use reconstruct_cli_metric.py to create it."
            )
        if self.distance_units != "meters":
            raise ValueError(f"Unsupported distance units: {self.distance_units}")
        self._validate_shapes()

    @property
    def view_count(self):
        return int(self.world_points.shape[0])

    def _validate_shapes(self):
        if self.world_points.ndim != 4 or self.world_points.shape[-1] != 3:
            raise ValueError(
                f"world_points must have shape (N, H, W, 3), got {self.world_points.shape}"
            )
        grid_shape = self.world_points.shape[:3]
        for name, array in (
            ("final_masks", self.final_masks),
            ("depth_z", self.depth_z),
            ("confidence", self.confidence),
        ):
            if array.shape != grid_shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match {grid_shape}"
                )
        if self.images.shape[:3] != grid_shape or self.images.shape[-1] not in {3, 4}:
            raise ValueError(
                "processed_images_uint8 must match the world-point pixel grid; "
                f"got {self.images.shape} versus {grid_shape}"
            )
        if len(self.image_names) != self.view_count:
            raise ValueError("image_names length does not match the number of views")
        if self.metric_scaling_factor.reshape(-1).size not in {1, self.view_count}:
            raise ValueError(
                "metric_scaling_factor must be scalar or have one value per view"
            )

    def image(self, view_index):
        return self.images[int(view_index), ..., :3].copy()

    def scale_for_view(self, view_index):
        values = self.metric_scaling_factor.reshape(-1)
        return float(values[0] if values.size == 1 else values[int(view_index)])

    def sample_point(self, view_index, x, y, radius):
        view_index = int(view_index)
        x = int(round(x))
        y = int(round(y))
        if not 0 <= view_index < self.view_count:
            raise ValueError(f"View index is outside the archive: {view_index}")
        height, width = self.final_masks.shape[1:]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point ({x}, {y}) is outside the {width}x{height} image")

        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        points_patch = self.world_points[view_index, y0:y1, x0:x1]
        depth_patch = self.depth_z[view_index, y0:y1, x0:x1]
        confidence_patch = self.confidence[view_index, y0:y1, x0:x1]
        valid = self.final_masks[view_index, y0:y1, x0:x1].copy()
        valid &= np.isfinite(points_patch).all(axis=-1)
        valid &= np.isfinite(depth_patch)
        valid &= depth_patch > 0
        if not valid.any():
            raise ValueError(
                f"No valid 3D samples in the patch around point ({x}, {y}) "
                f"in image {self.image_names[view_index]}"
            )

        patch_coordinates = np.argwhere(valid)
        candidate_points = points_patch[valid]
        candidate_depths = depth_patch[valid]
        candidate_confidence = confidence_patch[valid]

        if candidate_depths.size >= 3:
            median_depth = float(np.median(candidate_depths))
            mad = float(np.median(np.abs(candidate_depths - median_depth)))
            if mad > 0:
                tolerance = 3.0 * 1.4826 * mad
            else:
                tolerance = max(abs(median_depth) * 1e-4, 1e-5)
            depth_inliers = np.abs(candidate_depths - median_depth) <= tolerance
            if depth_inliers.any():
                candidate_points = candidate_points[depth_inliers]
                candidate_depths = candidate_depths[depth_inliers]
                candidate_confidence = candidate_confidence[depth_inliers]
                patch_coordinates = patch_coordinates[depth_inliers]

        robust_center = np.median(candidate_points, axis=0)
        selected_index = int(
            np.argmin(np.linalg.norm(candidate_points - robust_center, axis=1))
        )
        selected_point = candidate_points[selected_index]
        selected_yx = patch_coordinates[selected_index]
        finite_confidence = candidate_confidence[np.isfinite(candidate_confidence)]
        median_confidence = (
            float(np.median(finite_confidence))
            if finite_confidence.size
            else float("nan")
        )
        return {
            "point": selected_point.astype(np.float64),
            "depth_z_m": float(candidate_depths[selected_index]),
            "resolved_x": int(x0 + selected_yx[1]),
            "resolved_y": int(y0 + selected_yx[0]),
            "sample_count": int(candidate_points.shape[0]),
            "confidence_median": median_confidence,
        }


def view_index_from_label(label):
    if label is None:
        return 0
    return int(str(label).split(":", 1)[0])


def normalize_selected_points(points):
    normalized = []
    for point in points or []:
        if not isinstance(point, dict):
            raise ValueError("Invalid point state; clear the selection and try again")
        normalized.append(
            {
                "view_index": int(point["view_index"]),
                "x": int(point["x"]),
                "y": int(point["y"]),
            }
        )
    return normalized


def selection_status(dataset, points):
    points = normalize_selected_points(points)
    if not points:
        return "Kliknij punkt A na dowolnym zdjęciu."

    lines = []
    for index, point in enumerate(points):
        label = "AB"[index]
        image_name = dataset.image_names[point["view_index"]]
        lines.append(
            f"- Punkt {label}: **{image_name}**, piksel ({point['x']}, {point['y']})"
        )
    if len(points) == 1:
        lines.append(
            "\nWybierz zdjęcie zawierające drugi koniec odcinka i kliknij punkt B."
        )
    else:
        lines.append("\nOba punkty są gotowe do obliczenia odległości.")
    return "\n".join(lines)


def annotate_image(image, points, view_index):
    annotated = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    colors = [(255, 40, 40), (40, 220, 80)]
    for index, point in enumerate(normalize_selected_points(points)):
        if point["view_index"] != int(view_index):
            continue
        x, y = point["x"], point["y"]
        color = colors[index]
        radius = 6
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=3,
        )
        draw.text((x + 8, y - 8), "AB"[index], fill=color)
    return np.asarray(annotated)


def measure_selected_points(dataset, points, patch_radius):
    points = normalize_selected_points(points)
    if len(points) != 2:
        raise ValueError("Exactly two points are required")
    point_a_selection, point_b_selection = points
    sample_a = dataset.sample_point(
        point_a_selection["view_index"],
        point_a_selection["x"],
        point_a_selection["y"],
        patch_radius,
    )
    sample_b = dataset.sample_point(
        point_b_selection["view_index"],
        point_b_selection["x"],
        point_b_selection["y"],
        patch_radius,
    )
    predicted_distance_m = float(np.linalg.norm(sample_a["point"] - sample_b["point"]))
    return {
        "selection_a": point_a_selection,
        "selection_b": point_b_selection,
        "sample_a": sample_a,
        "sample_b": sample_b,
        "predicted_distance_m": predicted_distance_m,
        "cross_view": (
            point_a_selection["view_index"] != point_b_selection["view_index"]
        ),
    }


def append_measurement(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.is_file() and path.stat().st_size > 0
    if has_header:
        with path.open("r", encoding="utf-8", newline="") as input_file:
            existing_header = next(csv.reader(input_file), [])
        if existing_header == LEGACY_MEASUREMENT_FIELDS_V2:
            migrate_v2_measurement_csv(path)
            existing_header = MEASUREMENT_FIELDS
        if existing_header != MEASUREMENT_FIELDS:
            raise ValueError(
                f"CSV schema mismatch in {path}. Use a new --output path; the "
                "multi-view format cannot be appended to an older measurement file."
            )
    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MEASUREMENT_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def migrate_v2_measurement_csv(path):
    """Atomically add depth columns while preserving existing v2 measurements."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != LEGACY_MEASUREMENT_FIELDS_V2:
            raise ValueError(f"Cannot migrate unexpected CSV schema in {path}")
        rows = list(reader)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=MEASUREMENT_FIELDS)
            writer.writeheader()
            for existing_row in rows:
                existing_row["depth_z_a_m"] = ""
                existing_row["depth_z_b_m"] = ""
                writer.writerow(existing_row)
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_app(dataset, output_path, patch_radius):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            'Gradio is required. Install the project with: pip install -e ".[gradio]"'
        ) from error

    labels = [f"{index}: {name}" for index, name in enumerate(dataset.image_names)]

    def change_view(view_label, points):
        view_index = view_index_from_label(view_label)
        current = normalize_selected_points(points)
        return (
            annotate_image(dataset.image(view_index), current, view_index),
            current,
            selection_status(dataset, current),
        )

    def register_click(view_label, points, event: gr.SelectData):
        view_index = view_index_from_label(view_label)
        current = normalize_selected_points(points)
        if not isinstance(event.index, (tuple, list)) or len(event.index) < 2:
            raise gr.Error("Nie udało się odczytać współrzędnych kliknięcia.")
        x, y = int(round(event.index[0])), int(round(event.index[1]))
        if len(current) >= 2:
            current = []
        current.append({"view_index": view_index, "x": x, "y": y})
        return (
            annotate_image(dataset.image(view_index), current, view_index),
            current,
            selection_status(dataset, current),
        )

    def remove_last_point(view_label, points):
        view_index = view_index_from_label(view_label)
        current = normalize_selected_points(points)
        if current:
            current.pop()
        return (
            annotate_image(dataset.image(view_index), current, view_index),
            current,
            selection_status(dataset, current),
        )

    def reset_points(view_label):
        view_index = view_index_from_label(view_label)
        return dataset.image(view_index), [], selection_status(dataset, [])

    def calculate_and_save(
        points,
        scene_name,
        subset_label,
        segment_name,
        annotation_trial,
        reference_distance_m,
        reference_uncertainty_mm,
    ):
        if not str(segment_name or "").strip():
            raise gr.Error("Podaj nazwę mierzonego odcinka.")
        if reference_distance_m is None or float(reference_distance_m) <= 0:
            raise gr.Error("Odległość rzeczywista musi być większa od zera.")
        if reference_uncertainty_mm is None or float(reference_uncertainty_mm) < 0:
            raise gr.Error("Niepewność pomiaru nie może być ujemna.")

        try:
            measurement = measure_selected_points(dataset, points, patch_radius)
        except ValueError as error:
            raise gr.Error(str(error)) from error

        selection_a = measurement["selection_a"]
        selection_b = measurement["selection_b"]
        view_a = selection_a["view_index"]
        view_b = selection_b["view_index"]
        sample_a = measurement["sample_a"]
        sample_b = measurement["sample_b"]
        point_a = sample_a["point"]
        point_b = sample_b["point"]
        reference_distance_m = float(reference_distance_m)
        predicted_distance_m = measurement["predicted_distance_m"]
        signed_error_m = predicted_distance_m - reference_distance_m
        absolute_error_m = abs(signed_error_m)
        percentage_error = absolute_error_m / reference_distance_m * 100.0
        row = {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_archive": str(dataset.path),
            "scene_name": str(scene_name or dataset.scene_name).strip(),
            "subset_label": str(subset_label or dataset.subset_label).strip(),
            "view_count": dataset.view_count,
            "image_a_index": view_a,
            "image_a_name": dataset.image_names[view_a],
            "image_b_index": view_b,
            "image_b_name": dataset.image_names[view_b],
            "cross_view_measurement": measurement["cross_view"],
            "segment_name": str(segment_name).strip(),
            "annotation_trial": int(annotation_trial or 1),
            "reference_distance_m": reference_distance_m,
            "reference_uncertainty_mm": float(reference_uncertainty_mm),
            "predicted_distance_m": predicted_distance_m,
            "signed_error_m": signed_error_m,
            "absolute_error_m": absolute_error_m,
            "absolute_error_cm": absolute_error_m * 100.0,
            "percentage_error": percentage_error,
            "click_a_x": selection_a["x"],
            "click_a_y": selection_a["y"],
            "click_b_x": selection_b["x"],
            "click_b_y": selection_b["y"],
            "resolved_a_x": sample_a["resolved_x"],
            "resolved_a_y": sample_a["resolved_y"],
            "resolved_b_x": sample_b["resolved_x"],
            "resolved_b_y": sample_b["resolved_y"],
            "point_a_x_m": float(point_a[0]),
            "point_a_y_m": float(point_a[1]),
            "point_a_z_m": float(point_a[2]),
            "point_b_x_m": float(point_b[0]),
            "point_b_y_m": float(point_b[1]),
            "point_b_z_m": float(point_b[2]),
            "patch_radius_px": patch_radius,
            "valid_samples_a": sample_a["sample_count"],
            "valid_samples_b": sample_b["sample_count"],
            "confidence_a_median": sample_a["confidence_median"],
            "confidence_b_median": sample_b["confidence_median"],
            "metric_scaling_factor_a": dataset.scale_for_view(view_a),
            "metric_scaling_factor_b": dataset.scale_for_view(view_b),
            "depth_z_a_m": sample_a["depth_z_m"],
            "depth_z_b_m": sample_b["depth_z_m"],
            "sampling_method": "local_3d_medoid_after_depth_mad_filter",
        }
        try:
            append_measurement(output_path, row)
        except ValueError as error:
            raise gr.Error(str(error)) from error

        measurement_type = (
            "różne zdjęcia" if measurement["cross_view"] else "to samo zdjęcie"
        )
        return (
            f"### Wynik zapisany\n\n"
            f"- Punkt A: **{dataset.image_names[view_a]}**\n"
            f"- G\u0142\u0119boko\u015b\u0107 Z punktu A: **{sample_a['depth_z_m']:.4f} m**\n"
            f"- Punkt B: **{dataset.image_names[view_b]}**\n"
            f"- G\u0142\u0119boko\u015b\u0107 Z punktu B: **{sample_b['depth_z_m']:.4f} m**\n"
            f"- Tryb: **{measurement_type}**\n"
            f"- Rekonstrukcja: **{predicted_distance_m:.4f} m**\n"
            f"- Pomiar rzeczywisty: **{reference_distance_m:.4f} m**\n"
            f"- Błąd ze znakiem: **{signed_error_m:+.4f} m**\n"
            f"- Błąd bezwzględny: **{absolute_error_m * 100:.2f} cm**\n"
            f"- Błąd procentowy: **{percentage_error:.2f}%**\n"
            f"- Plik wynikowy: `{output_path}`"
        )

    with gr.Blocks(title="MapAnything — pomiar metryczny między widokami") as app:
        gr.Markdown(
            "# MapAnything — pomiar odległości między widokami\n"
            "Wybierz zdjęcie i kliknij punkt A. Następnie możesz zmienić zdjęcie "
            "i kliknąć punkt B. Oba punkty są odczytywane we wspólnym układzie 3D; "
            "skala nie jest dopasowywana do wartości referencyjnej."
        )
        point_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=2):
                view_dropdown = gr.Dropdown(
                    choices=labels,
                    value=labels[0],
                    label="Aktualnie wyświetlane zdjęcie",
                )
                image_component = gr.Image(
                    value=dataset.image(0),
                    type="numpy",
                    label="Obraz po preprocessingu — kliknij aktualny punkt",
                    interactive=False,
                )
                click_status = gr.Markdown(selection_status(dataset, []))
                with gr.Row():
                    remove_last_button = gr.Button("Usuń ostatni punkt")
                    reset_button = gr.Button("Wyczyść oba punkty")
            with gr.Column(scale=1):
                scene_input = gr.Textbox(value=dataset.scene_name, label="Scena")
                subset_input = gr.Textbox(value=dataset.subset_label, label="Wariant")
                segment_input = gr.Textbox(label="Nazwa odcinka")
                reference_input = gr.Number(label="Odległość rzeczywista [m]")
                uncertainty_input = gr.Number(
                    value=1.0,
                    minimum=0,
                    label="Niepewność pomiaru miarką [mm]",
                )
                trial_input = gr.Number(
                    value=1,
                    minimum=1,
                    precision=0,
                    label="Numer powtórzenia kliknięcia",
                )
                gr.Markdown(
                    f"Sąsiedztwo: **{2 * patch_radius + 1}×{2 * patch_radius + 1} px**. "
                    "Punkty poza maską poprawności są odrzucane."
                )
                save_button = gr.Button("Oblicz i zapisz", variant="primary")
                result_output = gr.Markdown()

        view_dropdown.change(
            change_view,
            inputs=[view_dropdown, point_state],
            outputs=[image_component, point_state, click_status],
        )
        image_component.select(
            register_click,
            inputs=[view_dropdown, point_state],
            outputs=[image_component, point_state, click_status],
        )
        remove_last_button.click(
            remove_last_point,
            inputs=[view_dropdown, point_state],
            outputs=[image_component, point_state, click_status],
        )
        reset_button.click(
            reset_points,
            inputs=[view_dropdown],
            outputs=[image_component, point_state, click_status],
        )
        save_button.click(
            calculate_and_save,
            inputs=[
                point_state,
                scene_input,
                subset_input,
                segment_input,
                trial_input,
                reference_input,
                uncertainty_input,
            ],
            outputs=[result_output],
        )
    return app


def main():
    args = parse_args()
    dataset = MetricPredictionSet(args.predictions)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else dataset.path.parent / "metric_measurements_multiview.csv"
    )
    print(f"Archive: {dataset.path}")
    print(f"Scene: {dataset.scene_name}")
    print(f"Subset: {dataset.subset_label}")
    print(f"Views: {dataset.view_count}")
    print(f"Processed image grid: {dataset.images.shape[2]}x{dataset.images.shape[1]}")
    print(f"Measurements CSV: {output_path}")
    if args.dry_run:
        return

    app = build_app(dataset, output_path, args.patch_radius)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
