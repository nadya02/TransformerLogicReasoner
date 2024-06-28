#!/bin/bash

# Open git bash in 'Bachelor Thesis' folder and run the script there
ontology_dir="MavenProject/ontologies/owlxml"
output_dir="D:/Uni/Year 3/Bachelor Thesis/LogicReasoner/Autoregressive model ontology dataset/data"

#!/bin/bash

# Open git bash in 'Bachelor Thesis' folder and run the script there
ontology_dir="MavenProject/ontologies/owlxml"
output_dir="D:/Uni/Year 3/Bachelor Thesis/LogicReasoner/Autoregressive model ontology dataset/data"

echo "Current working directory: $(pwd)"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Get a list of all ontology files
ontology_files=($(find "$ontology_dir" -type f -name "*.xml"))
echo "${#ontology_files[@]} ontology files found."

# Randomly shuffle the array and select the first 100
selected_files=($(shuf -e "${ontology_files[@]}" -n 100))

# Loop through the selected ontology files
for file_path in "${selected_files[@]}"; do
    echo "Processing $(basename "$file_path")"

    # Define a specific output directory for each ontology based on its name
    ontology_name=$(basename "$file_path" .owl.xml)
    specific_output_dir="$output_dir/${ontology_name}_proofs"

    # Create this specific output directory
    #mkdir -p "$specific_output_dir"

    # Execute the Java program using Maven
    #mvn -f "D:/Uni/Year 3/Bachelor Thesis/LogicReasoner/MavenProject" exec:java -Dexec.mainClass="owlapi.tutorial.ProofsFirst" -Dexec.args="$file_path" -Dexec.outputFile="$specific_output_dir"
    mvn -f "D:/Uni/Year 3/Bachelor Thesis/LogicReasoner/MavenProject" exec:java -Dexec.mainClass="owlapi.tutorial.ProofsFirst" -Dexec.args="$file_path"
    
    # Check if the Java program ran successfully
    if [ $? -eq 0 ]; then
        echo "Java program executed successfully for $ontology_name."
    else
        echo "Java program failed to execute for $ontology_name. Continuing with next ontology."
    fi
done