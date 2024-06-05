package owlapi.tutorial;


import de.tu_dresden.inf.lat.evee.general.data.exceptions.ParsingException;
import de.tu_dresden.inf.lat.evee.proofGenerators.ELKProofGenerator;
import de.tu_dresden.inf.lat.evee.proofGenerators.specializedGenerators.ESPGMinimalSize;
import de.tu_dresden.inf.lat.evee.proofs.data.Inference;
import de.tu_dresden.inf.lat.evee.proofs.data.Proof;
import de.tu_dresden.inf.lat.evee.proofs.data.exceptions.ProofGenerationException;
import de.tu_dresden.inf.lat.evee.proofs.data.exceptions.ProofGenerationFailedException;
import de.tu_dresden.inf.lat.evee.proofs.interfaces.IProof;
import de.tu_dresden.inf.lat.evee.proofs.interfaces.IProofGenerator;
import de.tu_dresden.inf.lat.evee.proofs.tools.ProofTools;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.*;
import de.tu_dresden.inf.lat.evee.proofs.json.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


import java.io.File;

public class ProofsFirst {
    public static void main(String[] args) throws OWLOntologyCreationException, ProofGenerationException, ParsingException, IOException {

        File ontologyFile = new File(args[0]);
        String ontologyBaseName = ontologyFile.getName().replaceFirst("[.][^.]+$", "");

        // Create the result directory in the current working directory
        String resultDirectory = Paths.get("D:/Uni/Year 3/Bachelor Thesis/Transformer/Autoregressive model ontology dataset/data", ontologyBaseName + "_proofs").toString();

//         Ensure the result directory exists
        Path resultPath = Paths.get(resultDirectory);
        if (!Files.exists(resultPath)) {
            Files.createDirectories(resultPath); // Create the directory if it doesn't exist
        }

        // Keeps track of the generated axioms so that we don't have duplicates
        Set<OWLAxiom> generatedAxioms = new HashSet<>();

        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(ontologyFile); // take ontology from argument list

        OWLReasonerFactory rf = new ElkReasonerFactory();
        OWLReasoner reasoner = rf.createReasoner(ontology);
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY);

        OWLDataFactory factory = manager.getOWLDataFactory();
        IProofGenerator<OWLAxiom, OWLOntology> proofGenerator = new ELKProofGenerator();
        proofGenerator.setOntology(ontology);

        Set<OWLClass> classes = ontology.getClassesInSignature();
        List<OWLClass> classList = new ArrayList<>(classes);
        int numberIterations = 500;
        //Seed the same??
        Random random = new Random(0); // set seed
        int duplicateAxiomsCount = 0;
        int proofsGenerated = 0;

        for(int i = 0; i < numberIterations; i++){
            // Get a random index from the range of the classes size
            int randomClassIndex = random.nextInt(classes.size());

            // we get the random class that we are going to use as a parent node
            OWLClass childClass = classList.get(randomClassIndex);
//            System.out.println(childClass);

            //WHAT IS NODE SET AND HOW IS IT DIFFERENT FROM SET????
            //In the documentation it says that it gives the union of the entities contained in the NOdes
            // in this NodeSet, does that mean we get all direct subclasses in a Set of the given childClass??
            Set<OWLClass> superClassNodes = reasoner.getSuperClasses(childClass,false).getFlattened();

            if (superClassNodes.isEmpty())
                System.out.println("The chosen node has no subclasses");
            else {
                List<OWLClass> superClassesList = new ArrayList<>(superClassNodes);
                int randomSuperClassIndex = random.nextInt(superClassesList.size());
                OWLClass parentClass = superClassesList.get(randomSuperClassIndex);
//                System.out.println(parentClass);

                OWLAxiom axiom = factory.getOWLSubClassOfAxiom(childClass, parentClass);
                // Check if the current axiom has been selected already
                if(!generatedAxioms.contains(axiom)){
                    generatedAxioms.add(axiom);
                    try{
                        proofsGenerated++;
                        IProof<OWLAxiom> proof = proofGenerator.getProof(axiom);
//                        System.out.println(proof);
                        Set<OWLAxiom> reachableAssertions = ProofTools.reachableAssertions(proof);
                        List<OWLAxiom> reachableAssertionsList = new ArrayList<>(reachableAssertions);
                        Inference<OWLAxiom> inference = new Inference(axiom, "Input", reachableAssertionsList);
                        IProof<OWLAxiom> input = new Proof<>(axiom, Collections.singleton(inference));

                        // Save the proofs in the new directory named after the ontology
                        JsonProofWriter.<OWLAxiom>getInstance().writeToFile(input, resultDirectory + "/Input" + i);
                        JsonProofWriter.<OWLAxiom>getInstance().writeToFile(proof, resultDirectory + "/Proof" + i);
                    } catch (ProofGenerationException ex){
                        System.out.println("A proof couldn't be found!");
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
                else {
                    duplicateAxiomsCount++;
                    System.out.println(String.format("Duplicate axiom number %d selected!", duplicateAxiomsCount));
                }

            }
        }
        System.out.println(String.format("Proofs generated %d for %s ontology!", proofsGenerated, ontologyBaseName));
    }
}
