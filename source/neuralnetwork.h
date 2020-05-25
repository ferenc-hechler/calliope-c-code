#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

class Vect;
class NN;


namespace nn {

	NN* getBrain();

    //% blockId=nn_fcnnfromjson
    //% block="Json Brain|string %json"
    //% shim=nn::fcnnfromjson
	void fcnnfromjson(const char *jsonNN);

	//% blockId=nn_initfcnn
	//% block="Init Brain|number %inputs|number[] %hidden|number %outputs"
	//% shim=nn::initfcnn
	void initfcnn(int inputs, Vect *hidden, int outputs);

	//% blockId=nn_train
	//% block="Train|number[] %input|number %expected_output"
	//% shim=nn::train
	float train(Vect *x, Vect *y);


	//% blockId=nn_predict
	//% block="Predict|"
	//% shim=nn::predict
	void predict(Vect *x, Vect *y);

}


#endif // NEURALNETWORK_H
