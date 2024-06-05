package owlapi.tutorial;

import org.eclipse.rdf4j.model.vocabulary.OWL;
import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;

import java.io.File;
import java.io.FileOutputStream;

public class OWLAPIFirst {
    public static void main(String[] args) {
        // create a new ontology
        //OWLOntologyManager -> handles creating, loading and saving ontologies
        OWLOntologyManager man = OWLManager.createOWLOntologyManager();
        IRI pizzaOntology = IRI.create("http://protege.stanford.edu/ontologies/pizza/pizza.owl");
        OWLOntology o;

        // Retrieves the OWLDataFactory associated with the ontology manager
        //This factory is use to create various OWL elements like classes, properties, and individuals
        OWLDataFactory df = man.getOWLDataFactory();

        try {
            // creates a new empty ontology - an ontology without any axioms and annotations
            o = man.loadOntology(pizzaOntology);
            OWLReasonerFactory rf = new ReasonerFactory();
            OWLReasoner r = rf.createReasoner(o);
            // classifies the ontology
            r.precomputeInferences(InferenceType.CLASS_HIERARCHY);
            r.getSubClasses(df.getOWLClass("http://www.co−ode.org/ontologies/pizza/pizza.owl#RealItalianPizza"), false).forEach(System.out::println);
            System.out.println(r);
        } catch(OWLOntologyCreationException e){
            e.printStackTrace();
        }


        //System.out.println(man.getOntologies().size());
        //System.out.println(man.ontologies().count()); //streams???

    }
}
