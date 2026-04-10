# `utils/file_utils.py`

## Purpose

`utils/file_utils.py` handles saving an uploaded file to the configured upload directory.

## Main Function

### `save_uploaded_file(file)`

This helper takes a Flask uploaded file object and saves it safely.

## Detailed Flow

1. Read the original uploaded filename.
2. Sanitize the filename using `secure_filename`.
3. Build the destination path inside `UPLOAD_FOLDER`.
4. Save the uploaded file to disk.
5. Return:
   - sanitized filename
   - absolute file path

## Why `secure_filename` Matters

User-provided filenames can contain unsafe characters or path traversal attempts. `secure_filename` reduces that risk by converting the filename into a safer form for filesystem use.

## Why This File Is Useful

Even though the function is small, it keeps file-saving concerns isolated from the route logic. That makes the route easier to read and gives one central place to extend upload behavior later.

## Related Files

- [utils/file_utils.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\utils\file_utils.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)
- [config.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\config.py)

