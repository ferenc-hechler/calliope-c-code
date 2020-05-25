
#include "neuralnets/NN.h"
#include "neuralnets/Vect.h"

#include "json/Parser.h"
#include "json/NNJsonParser.h"

#include "platform/utils.h"

namespace nn {

	static NN *brain = 0;

	NN* getBrain() {
		return brain;
	}

    //% blockId=nn_fcnnfromjson
    //% block="Json Brain|string %json"
    //% shim=nn::fcnnfromjson
	void fcnnfromjson(const char *jsonNN) {
	    log("creating FCNN from JSON");
		if (brain != 0) {
			delete brain;
			brain = 0;
		}
		NNJsonParser nnParser;
		Parser parser(&nnParser);

		parser.parse(jsonNN);
		brain = (NN*) nnParser.getResult();

		if (brain != 0) {
			brain->print();
		}
	    log("FCNN successfully created\r\n");
	}


	//% blockId=nn_initfcnn
	//% block="Init Brain|number %inputs|number[] %hidden|number %outputs"
	//% shim=nn::initfcnn
	void initfcnn(int inputs, Vect *hidden, int outputs) {

		if (brain != 0) {
			delete brain;
		}
		brain = new NN(inputs);

	    int numHidden = hidden->getLength();
	    log("creating FCNN: in:");
	    logInt(inputs);
	    log("hidden-layers:");
	    hidden->print();
	    log("out:");
	    logInt(outputs);
	    for (int i=0; i<numHidden; i++) {
		    int nodes = hidden->get(i);
			brain->addLayer(nodes);
		    log("    hidden layer #");
		    logInt(i);
		    log(": ");
		    logInt(nodes);
	    }

		brain->addLayer(outputs);

	    log("FCNN successfully created\r\n");
	}


	//% blockId=nn_train
	//% block="Train|number[] %input|number %expected_output"
	//% shim=nn::train
	float train(Vect *x, Vect *y) {
		float learning_rate = 0.001;
		Vect *y_hat = brain->forwardPropagate(x);
		Vect *e = brain->backwardPropagate(y, y_hat, learning_rate);
		y_hat->sub(y);
		y_hat->sqr();
		float sum_sq_err = y_hat->sum();
		delete y_hat;
		delete e;
		return sum_sq_err;
	}


	//% blockId=nn_predict
	//% block="Predict|"
	//% shim=nn::predict
	void predict(Vect *x, Vect *y) {
		Vect *y_hat = brain->forwardPropagate(x);
		for (int i=0; i<y_hat->getLength(); i++) {
			y->set(i, y_hat->get(i));
		}
		delete y_hat;
	}



}
