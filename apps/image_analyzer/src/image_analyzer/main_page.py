
from io import BytesIO
from PIL import Image
from image_analyzer.api import get_image_ranking
import streamlit as st
from base64 import b64decode, b64encode

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# TODO: clean up, split the code better up.

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data: bytes = uploaded_file.getvalue()
    # TODO: base64 encode the bytes
    image_base64_encoded: str = b64encode(bytes_data).decode("utf-8")

    response: dict = get_image_ranking(image_base64_encoded)

    st.write("Image Ranking Response:")
    st.json(response)

    results = response.get("results", [])

    for res in results:
        person: str | None = res.get("image_base64")
        if not person:
            st.write("No person image found in the response.")
            continue

        # TODO: convert person base64 to image then display it
        if person:
            image = b64decode(person)
            image = Image.open(BytesIO(image))
            st.image(image, caption="Person Image", use_container_width=True)

        lower_body_res: str | None = res.get("lower_body", {})
        lower_body_base64: str | None = lower_body_res.get("image_base64")
        lower_body_ranking: float | None = lower_body_res.get("ranking")

        if lower_body_base64:
            lower_body = b64decode(lower_body_base64)
            image_lower_body = Image.open(BytesIO(lower_body))
            st.image(image_lower_body, caption="Lower body image", use_container_width=True)


