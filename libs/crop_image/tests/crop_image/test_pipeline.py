from re import I
import pytest
from unittest.mock import Mock
from PIL import Image
from crop_image.models import ImageParts
from crop_image.pipeline import pipeline
from ultralytics import YOLO


@pytest.fixture
def yolo_model(datadir):
    """Fixture to provide a YOLO model instance."""
    return YOLO(
        datadir / "yolov8n.pt"
    )  # Replace with the correct YOLO model path if needed


@pytest.fixture
def mock_yolo_model():
    """Fixture to provide a mock YOLO model."""
    return Mock(spec=YOLO)


@pytest.fixture
def sample_image():
    """Fixture to provide a sample PIL image."""
    return Image.new("RGB", (500, 500))


@pytest.mark.parametrize(
    "crop_person_return, is_sharp_return, has_face_return, expected_result_length",
    [
        # Case 1: All steps pass
        ([Image.new("RGB", (100, 100))], True, True, 1),
        # Case 2: No persons detected
        ([], None, None, 0),
        # Case 3: Cropped image is not sharp
        ([Image.new("RGB", (100, 100))], False, None, 0),
        # Case 4: Cropped image does not contain a face
        ([Image.new("RGB", (100, 100))], True, False, 0),
    ],
)
def test_pipeline(
    sample_image,
    mock_yolo_model,
    crop_person_return,
    is_sharp_return,
    has_face_return,
    expected_result_length,
):
    """Test pipeline with parameterized cases."""
    # Mock dependencies
    mock_crop_person_fn = Mock(return_value=crop_person_return)
    mock_is_sharp_fn = Mock(
        return_value=is_sharp_return if is_sharp_return is not None else False
    )
    mock_has_face_fn = Mock(
        return_value=has_face_return if has_face_return is not None else False
    )

    # Run the pipeline
    result = pipeline(
        image=sample_image,
        crop_model=mock_yolo_model,
        crop_person_fn=mock_crop_person_fn,
        is_sharp_fn=mock_is_sharp_fn,
        has_face_fn=mock_has_face_fn,
    )

    # Assertions
    assert len(result) == expected_result_length
    mock_crop_person_fn.assert_called_once_with(sample_image, mock_yolo_model, 0.5)
    if crop_person_return:
        mock_is_sharp_fn.assert_called()
        if is_sharp_return:
            mock_has_face_fn.assert_called()
        else:
            mock_has_face_fn.assert_not_called()
    else:
        mock_is_sharp_fn.assert_not_called()
        mock_has_face_fn.assert_not_called()

        mock_has_face_fn.assert_not_called()


def test_pipeline_with_real_image(yolo_model, datadir):
    """Test pipeline using the real example image."""
    # Get the path to the test image using pytest-datadir
    test_image_path = datadir / "chatgpt_man_running.jpg"

    # Load the test image as a PIL Image
    test_image = Image.open(test_image_path)

    # Run the pipeline with default methods
    result: list[ImageParts] = pipeline(image=test_image, crop_model=yolo_model)

    # Assertions
    assert len(result) > 0, "Pipeline should detect at least one valid cropped image"
    for cropped_image in result:
        assert isinstance(cropped_image, ImageParts), (
            "Result should contain ImageParts objects"
        )
