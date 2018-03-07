for (var i = 0; i < 10; i++) {
    console.log("Hello World");
}

var i = 0;
var text = "";
while (i < 10) {
    text += "The number is " + i;
    i++;
}

do {
    text += "The number is " + i;
    i++;
}
while (i < 10);