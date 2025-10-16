import os
import shutil
import argparse

def create_new_scenario(template_folder, new_folder_name):
    if not os.path.exists(template_folder):
        print(f"Template folder '{template_folder}' does not exist.")
        return

    # Create the new folder
    if os.path.exists(new_folder_name):
        print(f"Folder '{new_folder_name}' already exists.")
        return
    shutil.copytree(template_folder, new_folder_name)

    # Define replacements: filenames and class names
    replacements = {
        "scenario_template": new_folder_name.lower(),
        "ScenarioTemplate": new_folder_name.capitalize(),
        "ScenarioTemplateExtension": f"{new_folder_name.capitalize()}Extension"
    }

    # Rename files and update contents
    for root, dirs, files in os.walk(new_folder_name):
        for file in files:
            old_path = os.path.join(root, file)

            # Rename file if it contains 'scenario_template'
            if "scenario_template" in file:
                new_file_name = file.replace("scenario_template", new_folder_name.lower())
                new_path = os.path.join(root, new_file_name)
                os.rename(old_path, new_path)
                old_path = new_path

            # Replace class names and references inside the file
            with open(old_path, "r", encoding="utf-8") as f:
                content = f.read()

            for old, new in replacements.items():
                content = content.replace(old, new)

            with open(old_path, "w", encoding="utf-8") as f:
                f.write(content)

    print(f"New scenario folder '{new_folder_name}' created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new scenario folder from a template.")
    parser.add_argument("new_folder_name", help="The name of the new scenario folder to create")
    args = parser.parse_args()

    create_new_scenario("scenario_template", args.new_folder_name)
