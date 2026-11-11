import argparse
from datetime import datetime, timedelta
import os
import urllib.request
import urllib.error
import sys
import tarfile
import csv
import io

BASE_URL = "https://subwaydata.nyc/data/subwaydatanyc_{date}_csv.tar.xz"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")

def process_and_extract_tar_xz(file_path: str, date_str: str, output_dir: str):
    """
    Extract CSVs from a .tar.xz archive, add a 'day' column to each row,
    and save the extracted data directly to the output directory.
    """
    if not tarfile.is_tarfile(file_path):
        print(f"Error: {file_path} is not a valid tar archive.", file=sys.stderr)
        return

    with tarfile.open(file_path, "r:xz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue

            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue

            # Read CSV content from memory stream
            csv_text = io.TextIOWrapper(extracted_file, encoding="utf-8")
            reader = csv.reader(csv_text)

            try:
                header = next(reader)
            except StopIteration:
                continue  # Empty file

            # Append 'day' column header
            new_header = header + ["day"]

            # Construct output file path in the output directory
            base_filename = os.path.basename(member.name)
            out_csv_path = os.path.join(output_dir, f"processed_{base_filename}")

            with open(out_csv_path, "w", newline="", encoding="utf-8") as out_file:
                writer = csv.writer(out_file)
                writer.writerow(new_header)

                # Write rows with the added date string
                for row in reader:
                    writer.writerow(row + [date_str])

            print(f"Extracted and annotated: {out_csv_path}")

def download_range(start_date: datetime, end_date: datetime, output_dir: str, keep_archives: bool):
    """Download and process Subway Data NYC archives for each day in range."""
    if start_date > end_date:
        print("Error: Start date must be before or equal to end date.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = BASE_URL.format(date=date_str)
        archive_filename = f"subwaydatanyc_{date_str}_csv.tar.xz"
        archive_path = os.path.join(output_dir, archive_filename)

        print(f"Downloading: {url} -> {archive_path}")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req) as response, open(archive_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"Successfully saved archive: {archive_path}")

            # Extract archive and add the date column
            process_and_extract_tar_xz(archive_path, date_str, output_dir)

            # Cleanup archive if requested
            if not keep_archives and os.path.exists(archive_path):
                os.remove(archive_path)

        except urllib.error.HTTPError as e:
            print(f"Failed to download {date_str} (HTTP {e.code}): {e.reason}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {date_str}: {e}", file=sys.stderr)

        current_date += timedelta(days=1)

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract NYC subway data archives within a date range."
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=True,
        help="Start date in YYYY-MM-DD format (inclusive)"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=True,
        help="End date in YYYY-MM-DD format (inclusive)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./downloads",
        help="Directory where extracted CSV files will be saved (default: ./downloads)"
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the original downloaded .tar.xz archive files after extraction"
    )

    args = parser.parse_args()
    download_range(args.start_date, args.end_date, args.output_dir, args.keep_archives)

if __name__ == "__main__":
    main()