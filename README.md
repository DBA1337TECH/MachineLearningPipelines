## Machine Learning Aided Steganography
The goal of this project was more of generalizing on a modest dataset, and availble publicly, the use of audio to hide data within WAV files.

**Current State of Project:** Active Development and fine tuning to get to work, currently it's only a rough prototype code.

#### Discalimer
This isn't really working perfectly and is only a short idea, not intended for real world use, yet...


###  What and Why
The idea was first to first engineer a cryptographic system that could be hidden in plain sight.  Then it morphed to starting if I could hide sound files within a sound file.
Which ultimately led to engineering an encoder and decoder dual set.  Why I set out to do this, was is a series of code I write for freedom of speech and freedom to protect my own speech
in a manner of my choosing, not anyone else's decesion to decide when I can or how I can have private expressions of speech.
Whether that be an actor state foreign or domestic, cannot and should not have an authority that grants them a way to circumvent the 4th, 5th, or 1st amendments of the U.S. constitution.

Therefore I have over the course of 3 years developed and slowly making some of those ideas and technologies publicly avaiable in hopes they will be morphed, used, and ultimately
made better to promote freedom of speech while maintaining a peace of mind that the speech that you create, express, and/or produce are logarithmically hard to decipher unless read by the intended audience and
receipient.

### Encoder
The encoder will attempt to merge two audio files, where their is the true audio and the visible audio.  The true audio is a message you'd like to hide.
The visible audio will be the decoy message that hopefully will be a distraction to someone intercepting say a VOIP call.


### Decoder
the Decoder will take in the mixed audio (true + visible) audio channels and extract the true message.

### Use Case
Communications that are often 

