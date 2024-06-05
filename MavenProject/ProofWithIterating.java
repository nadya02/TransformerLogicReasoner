package owlapi.tutorial;


import de.tu_dresden.inf.lat.evee.proofGenerators.ELKProofGenerator;
import de.tu_dresden.inf.lat.evee.proofGenerators.specializedGenerators.ESPGMinimalSize;
import de.tu_dresden.inf.lat.evee.proofs.data.exceptions.ProofGenerationException;
import de.tu_dresden.inf.lat.evee.proofs.data.exceptions.ProofGenerationFailedException;
import de.tu_dresden.inf.lat.evee.proofs.interfaces.IProof;
import de.tu_dresden.inf.lat.evee.proofs.interfaces.IProofGenerator;
import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.Node;
import org.semanticweb.owlapi.reasoner.NodeSet;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import de.tu_dresden.inf.lat.evee.proofs.json.*;

import java.io.File;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;


import java.io.File;

public class ProofsWithIterating {
    public static void main(String[] args) throws OWLOntologyCreationException, ProofGenerationException {

        /*// Load the ontology
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(new File("goslim_agr.owl"));

        // Computes the entire class hierarchy
        OWLReasoner reasoner = new ReasonerFactory().createReasoner(ontology);
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY);

        IProofGenerator<OWLAxiom, OWLOntology> proofGenerator = new ESPGMinimalSize();
        proofGenerator.setOntology(ontology);

        /*

        NodeSet<OWLClass> subClassNodes = reasoner.getSubClasses(topClass);

        Node<OWLClass> subClassNode = subClassNodes.getNodes().iterator().next();
        OWLClass subClass = subClassNode.getEntities().iterator().next();

        // generate a sub class axiom (represents the link in the hierarchy
        OWLDataFactory factory = manager.getOWLDataFactory();
        OWLAxiom axiom = factory.getOWLSubClassOfAxiom(topClass,subClass);
        */
        // create the proof generator
        IProofGenerator<OWLAxiom, OWLOntology> proofGenerator = new ESPGMinimalSize();
        proofGenerator.setOntology(ontology);

        JsonAxiom2StringConverter axiomConverter = new JsonAxiom2StringConverter();

        int failedProofs = 0;

        Node<OWLClass> topNode = reasoner.getTopClassNode();
        //OWLClass topClass = topNode.getEntities().iterator().next();

        //NodeSet<OWLClass> subClassNodes = reasoner.getSubClasses(topClass);

        Queue<Node<OWLClass>> queue = new LinkedList<>();
        Set<Node<OWLClass>> visited = new HashSet<>();
        queue.add(topNode);
        while(!queue.isEmpty()){
            Node<OWLClass> currentNode = queue.poll();
            for(OWLClass currentClass : currentNode.getEntities()) {
                OWLClass currentSubClass = currentClass;
                NodeSet<OWLClass> subClasses = reasoner.getSubClasses(currentSubClass);
                //getFlattened?
                for (OWLClass child : subClasses.getFlattened()){

                }
            }


            for(Node<OWLClass> subClass: reasoner.getSubClasses(currentSubClass)){

            }
        }
    }
}
