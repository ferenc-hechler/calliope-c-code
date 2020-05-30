/**
 * Calliope ML Racer
 *
 * Based ont the simple template for use with Calliope mini.
 * @copyright (c) Calliope gGmbH.
 * @author Matthias L. Jugel.
 * Licensed under the Apache License 2.0
 */

#include <MicroBit.h>

#include "neuralnetwork.h"
#include "platform/Utils.h"
#include "neuralnets/Vect.h"
#include "neuralnets/NN.h"
#include "neuralnets/NNUtils.h"

MicroBit uBit;



void greeting() {
    uBit.serial.baud(115200);
    uBit.serial.send("Calliope mini template v1.0\r\n");
    uBit.serial.send("Hello world!\r\n");
}

void run() {
	Vect hidden = createVect(2, 10.0f, 10.0f);

	nn::initfcnn(3, &hidden, 2);

	Vect x = createVect(3, 2.0f, 3.0f ,4.0f);
	Vect y = createVect(2, 1.0f, 0.0f);

	float err = nn::train(&x, &y);
	logFloat(err);

	nn::predict(&x, &y);
	logFloat(y.get(0));
	log(", ");
	logFloat(y.get(1));

	nn::getBrain()->print();

	log("\r\nfinished");
}




int main(void) {
    uBit.init();

    greeting();
    run();

    log("\r\nfinished\r\n");
    return 0;
}
