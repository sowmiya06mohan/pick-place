from flask import Flask, render_template, request, send_from_directory
import os
import subprocess
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    result_video = None
    error = None

    if request.method == "POST":

        if "video" not in request.files:
            error = "Please select a video."
            return render_template(
                "index.html",
                result_video=result_video,
                error=error
            )

        video = request.files["video"]

        if video.filename == "":
            error = "Please select a video."
            return render_template(
                "index.html",
                result_video=result_video,
                error=error
            )

        filename = f"{uuid.uuid4()}_{video.filename}"
        input_path = os.path.join(UPLOAD_FOLDER, filename)

        video.save(input_path)

        try:
            subprocess.run(
                [
                    "python",
                    "src/pick_place_pipeline.py",
                    input_path
                ],
                check=True
            )

            output_file = "final_pick_place.mp4"

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_file
            )

            if os.path.exists(output_path):
                result_video = output_file
            else:
                error = "Processing completed, but output video was not found."

        except subprocess.CalledProcessError:
            error = "An error occurred while processing the video."

    return render_template(
        "index.html",
        result_video=result_video,
        error=error
    )


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )