# Glitchnet

This Tensorflow-based network is the result of my 'final project' for [Kadenze](https://www.kadenze.com)'s [Creative Applications of Deep Learning](https://github.com/pkmital/CADL) course.

I was experimenting with using an LSTM recurrent neural network to predict images given an entire or partial image. While playing around, I noticed that because my network was outputing 1-dimensional arrays, the resulting images had interesting horizontal artifacts. When I trained the network on just a few images, the result was a very glitchy version of the original image. By adding some randomness with dropout and converting multiple predictions into a GIF, I could create an interesting 'glitched' version of my original image.

![glitchnet](http://i.giphy.com/l3q2vgq6psmq5LbUI.gif)

The resulting image was a little hard to understand, but I thought it looked cool. By training the network for just a few more minutes, it was able to output much more accurate images. So if the first network created a 'glitch' effect, this network had a 'pixelate' effect. I think it kind of looks like what you'd seen on an original gameboy.

![pixelatenet](http://i.giphy.com/26gspVChlTCjHj9K0.gif)
