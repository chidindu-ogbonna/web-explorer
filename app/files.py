import base64
import io
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

import cv2
import numpy as np
from browser_use.agent.views import AgentHistoryList
from PIL import Image, ImageDraw, ImageFont
from pydantic import HttpUrl, TypeAdapter

from app.logger import base_logger

logger = base_logger.getChild(__name__)


class Font(TypedDict):
    regular: ImageFont.ImageFont | ImageFont.FreeTypeFont
    title: ImageFont.ImageFont | ImageFont.FreeTypeFont
    goal: ImageFont.ImageFont | ImageFont.FreeTypeFont


class GIFParams(TypedDict, total=False):
    duration: int
    regular_font_size: int
    title_font_size: int
    goal_font_size: int
    margin: int
    line_spacing: float
    title_text: str
    use_logo: bool


class HistoryMedia(TypedDict):
    gif: str | None
    mp4: str | None


class FileUtils:
    @staticmethod
    def _pil_to_cv2(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to CV2 format."""
        # Convert PIL image to RGB if it's not
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert to numpy array and swap RGB to BGR
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _load_font(
        *,
        regular_font_size: int = 17,
        title_font_size: int = 22,
        goal_font_size: int = 17,
    ) -> Font:
        regular_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        goal_font = regular_font

        try:
            font_options = ["Helvetica", "Arial", "DejaVuSans", "Verdana"]  # in order of preference
            font_loaded = False

            for font_name in font_options:
                try:
                    regular_font = ImageFont.truetype(font_name, regular_font_size)
                    title_font = ImageFont.truetype(font_name, title_font_size)
                    goal_font = ImageFont.truetype(font_name, goal_font_size)
                    font_loaded = True
                    break
                except OSError:
                    continue

            if not font_loaded:
                msg = "No preferred fonts found"
                raise OSError(msg)
        except OSError:
            regular_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            goal_font = regular_font
        return Font(regular=regular_font, title=title_font, goal=goal_font)

    @staticmethod
    def _wrap_text(*, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> str:
        """Wrap text to fit within a given width."""
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            line = " ".join(current_line)
            bbox = font.getbbox(line)
            if bbox[2] > max_width:
                if len(current_line) == 1:
                    lines.append(current_line.pop())
                else:
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    @staticmethod
    def _create_frame(
        *,
        text: str,
        screenshot: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        logo: Image.Image | None,
        line_spacing: float = 1.3,
    ) -> Image.Image:
        img_bytes = FileUtils.decode_image_from_base64(screenshot)
        template = Image.open(io.BytesIO(img_bytes))
        image = Image.new("RGB", template.size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        # # Calculate vertical center of image
        center_y = image.height // 2
        # Draw task text with increased font size
        margin = 140  # Increased margin
        max_width = image.width - (2 * margin)
        # increase the font size by 16
        larger_font = ImageFont.truetype(font.path, font.size + 16)  # pyright: ignore[reportAttributeAccessIssue]
        wrapped_text = FileUtils._wrap_text(text=text, font=larger_font, max_width=max_width)
        # Calculate line height with spacing
        line_height = larger_font.size * line_spacing

        # Split text into lines and draw with custom spacing
        lines = wrapped_text.split("\n")
        total_height = line_height * len(lines)

        # Start position for first line
        text_y = center_y - (total_height / 2) + 50  # Shifted down slightly
        for line in lines:
            # Get line width for centering
            line_bbox = draw.textbbox((0, 0), line, font=larger_font)
            text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2
            draw.text((text_x, text_y), line, font=larger_font, fill=(255, 255, 255))
            text_y += line_height

        # Add logo if provided (top right corner)
        if logo:
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            image.paste(logo, (logo_x, logo_margin), logo if logo.mode == "RGBA" else None)

        return image

    @staticmethod
    def _add_overlay_to_image(  # noqa: PLR0913
        *,
        image: Image.Image,
        step_number: int,
        goal_text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        margin: int,
        logo: Image.Image | None = None,
    ) -> Image.Image:
        """Add step number and goal overlay to an image."""
        image = image.convert("RGBA")
        txt_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Add step number (bottom left)
        step_text = str(step_number)
        step_bbox = draw.textbbox((0, 0), step_text, font=font)
        step_width = step_bbox[2] - step_bbox[0]
        step_height = step_bbox[3] - step_bbox[1]

        # Position step number in bottom left
        x_step = margin + 10  # Slight additional offset from edge
        y_step = image.height - margin - step_height - 10  # Slight offset from bottom

        # Draw rounded rectangle background for step number
        padding = 20  # Increased padding
        step_bg_bbox = (
            x_step - padding,
            y_step - padding,
            x_step + step_width + padding,
            y_step + step_height + padding,
        )
        draw.rounded_rectangle(step_bg_bbox, radius=15, fill=(255, 255, 255, 180))  # White with alpha
        # Draw step number
        draw.text((x_step, y_step), step_text, font=font, fill=(0, 0, 0, 255))  # Black text

        # Draw goal text (centered, bottom)
        max_width = image.width - (4 * margin)
        wrapped_goal = FileUtils._wrap_text(text=goal_text, font=font, max_width=max_width)
        goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=font)
        goal_width = goal_bbox[2] - goal_bbox[0]
        goal_height = goal_bbox[3] - goal_bbox[1]

        # Center goal text horizontally, place above step number
        x_goal = (image.width - goal_width) // 2
        y_goal = y_step - goal_height - padding * 2  # More space between step and goal

        # Draw rounded rectangle background for goal
        padding_goal = 20  # Increased padding for goal
        goal_bg_bbox = (
            x_goal - padding_goal,
            y_goal - padding_goal,
            x_goal + goal_width + padding_goal,
            y_goal + goal_height + padding_goal,
        )
        draw.rounded_rectangle(goal_bg_bbox, radius=15, fill=(255, 255, 255, 100))  # White with alpha

        # Draw goal text
        draw.multiline_text((x_goal, y_goal), wrapped_goal, font=font, fill=(0, 0, 0, 255), align="center")

        # Add logo if provided (top right corner)
        if logo:
            logo_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == "RGBA" else None)
            txt_layer = Image.alpha_composite(logo_layer, txt_layer)

        # Composite and convert
        result = Image.alpha_composite(image, txt_layer)
        return result.convert("RGB")

    @staticmethod
    def _get_logo() -> Image.Image | None:
        try:
            # TODO(promise): Set logo
            logo = Image.open("./static/mployee.png")
            logo_height = 150
            aspect_ratio = logo.width / logo.height
            logo_width = int(logo_height * aspect_ratio)
            return logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        except Exception:
            logger.exception("Failed to load logo")
            return None

    # Public methods

    @staticmethod
    def is_valid_url(url: str | None | object) -> bool:
        if url is None:
            return False

        if not isinstance(url, str):
            return False

        # Prepend 'http://' if no scheme is present
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        url_validator = TypeAdapter(HttpUrl)
        try:
            url_validator.validate_python(url)
            domain = url.split("//")[1].split("/")[0]
            if "." not in domain:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def encode_image_to_base64(image: bytes) -> str:
        """Encode image bytes to a base64 encoded string."""
        return base64.b64encode(image).decode()

    @staticmethod
    def decode_image_from_base64(string: str) -> bytes:
        """Decode a base64 encoded image string to bytes."""
        return base64.b64decode(string)

    @staticmethod
    def create_video_from_images(
        *,
        images: list[Image.Image],
        output_path: str,
        fps: int = 1,
    ) -> str | None:
        try:
            if not images:
                logger.warning("No images provided to create video")
                return None
            # Get dimensions from first image
            height, width = np.array(images[0]).shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter.fourcc(*"avc1")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write each frame
            for image in images:
                cv2_image = FileUtils._pil_to_cv2(image)
                out.write(cv2_image)
            out.release()
        except Exception:
            logger.exception("Failed to create video")
            return None
        return output_path

    @staticmethod
    def create_gif_from_images(
        *,
        images: list[Image.Image],
        output_path: str,
        duration: int,
    ) -> str:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        return output_path

    @staticmethod
    def create_media_from_history_list(  # noqa: C901
        *,
        history_list: AgentHistoryList,
        screenshots_to_append: list[str | bytes],
        filename: str,
        output_format: list[Literal["gif", "mp4"]],
        params: GIFParams | None = None,
    ) -> HistoryMedia | None:
        images = []

        try:
            gif_params = GIFParams(
                duration=5000,
                regular_font_size=17,
                title_font_size=22,
                goal_font_size=17,
                margin=10,
                line_spacing=1.3,
                title_text="Task Run",
                use_logo=False,
            )
            if params:
                gif_params.update(params)

            logo: Image.Image | None = None
            if gif_params.get("use_logo", False):
                logo = FileUtils._get_logo()

            font = FileUtils._load_font(
                regular_font_size=gif_params["regular_font_size"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                title_font_size=gif_params["title_font_size"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                goal_font_size=gif_params["goal_font_size"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
            )
            images: list[Image.Image] = []
            images.append(
                FileUtils._create_frame(
                    text=gif_params["title_text"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    screenshot=history_list.history[0].state.screenshot or "",
                    font=font["title"],
                    line_spacing=gif_params["line_spacing"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    logo=logo,
                ),
            )
            for i, item in enumerate(history_list.history, 1):
                if not item.state.screenshot:
                    continue
                img_bytes = FileUtils.decode_image_from_base64(item.state.screenshot)
                image = Image.open(io.BytesIO(img_bytes))
                if item.model_output:
                    image = FileUtils._add_overlay_to_image(
                        image=image,
                        step_number=i,
                        goal_text=item.model_output.current_state.next_goal or "",
                        font=font["title"],
                        margin=gif_params["margin"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        logo=logo,
                    )
                images.append(image)
            for screenshot in screenshots_to_append:
                images.append(
                    FileUtils._create_frame(
                        text="",
                        screenshot=FileUtils.encode_image_to_base64(screenshot)
                        if isinstance(screenshot, bytes)
                        else screenshot,
                        font=font["title"],
                        line_spacing=gif_params["line_spacing"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        logo=logo,
                    ),
                )
            if images:
                gif_path: str | None = None
                video_path: str | None = None

                if "gif" in output_format:
                    gif_path = FileUtils.create_gif_from_images(
                        images=images,
                        output_path=f"{filename}.gif",
                        duration=gif_params["duration"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    )
                if "mp4" in output_format:
                    video_path = FileUtils.create_video_from_images(images=images, output_path=f"{filename}.mp4", fps=1)
                return HistoryMedia(gif=gif_path, mp4=video_path)
        except Exception:
            logger.exception("Failed to create GIF")
        return None

    @staticmethod
    def read_json_file(path: str) -> Any | None:  # noqa: ANN401
        filepath = Path(path)
        try:
            if filepath.exists():
                return json.loads(filepath.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read JSON file")
        return None

    @staticmethod
    def read_image_from_file(path: str, *, return_base64: bool = False) -> bytes | str | None:
        filepath = Path(path)
        try:
            with filepath.open("rb") as image_file:
                image_bytes = image_file.read()
                if return_base64:
                    return FileUtils.encode_image_to_base64(image_bytes)
                return image_bytes
        except Exception:
            logger.exception("Failed to read image file")
        return None

    @staticmethod
    def write_data_to_file(path: str, data: Any) -> None:  # noqa: ANN401
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    # history_media = FileUtils.create_media_from_history_list(
    #     history_list=AgentHistoryList.load_from_file(
    #         "/Users/6ones/Projects/mployee/backend/snippets/agent/history.json",
    #         AgentOutput,
    #     ),
    #     screenshots_to_append=[],
    #     filename="test_file_history",
    #     output_format=["mp4", "gif"],
    # )
    # print(history_media)

    # cookies = FileUtils.read_json_file("/Users/6ones/Projects/mployee/logs/cookies_tal_bd91871e.json")
    # print(cookies)
    # data = [
    #     {
    #         "name": "AEC",
    #         "path": "/",
    #         "value": "AZ6Zc-Vpllj-xz5W5qyLR5ndxZTNSqHNS699dpPhEM0uWQ6bT67biG1j85s",
    #         "domain": ".google.com",
    #         "secure": True,
    #         "expires": 1752254503.771263,
    #         "httpOnly": True,
    #         "sameSite": "Lax",
    #     },
    # ]
    # cookies_file = FileUtils.write_data_to_file("test-cookies.json", data)
    # print(cookies_file)
    pass
