import re


def clean_str(string):
    # copy pasted from Hollenstein's code...
    # TO-DO: Convert this to regex
    string = string.replace(".", "")
    string = string.replace(",", "")
    string = string.replace("--", "")
    string = string.replace("`", "")
    string = string.replace("''", "")
    string = string.replace("' ", " ")
    string = string.replace("*", "")
    string = string.replace("\\", "")
    string = string.replace(";", "")
    string = string.replace("- ", " ")
    string = string.replace("/", "-")
    string = string.replace("!", "")
    string = string.replace("?", "")

    # added by chip
    string = re.sub(r"[():]", "", string)
    string = re.sub(r"-$", "", string)

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()
