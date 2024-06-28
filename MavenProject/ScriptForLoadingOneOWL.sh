#!/bin/bash

ontology_dir="ontologies/owlxml/"
file_name="plio.protein-ligand-interaction-ontology.1.owl.xml"

specific_output_dir="${file_name}_proofs"
mkdir -p "$specific_output_dir"

mvn exec:java -Dexec.mainClass="owlapi.tutorial.ProofsFirst" -Dexec.args="$ontology_dir$file_name"