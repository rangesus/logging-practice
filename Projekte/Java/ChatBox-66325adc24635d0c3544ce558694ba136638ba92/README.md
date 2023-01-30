# ChatBox

ChatBox is written in Java.  It is named after the first name given to the Java programming language before it became "Java".
 

# Requirements

* Java 1.8
* [Maven] (for building)

# Build Instructions

ToChatBox the project, run the command below.

`mvn package`

This command will build the project and package it into an executable, shaded JAR. A shaded JAR file contains all of the project's dependencies. The shaded JAR file is saved here: `target/Chatbox-VERSION.jar`.

# Deploy Instructions

1. Copy the following files to the servChatBoxt them in theChatBoxdirectory:
   1. `target/ChatBox-VERSION.jar`: The executable, shaded JAR file that contains ChatBox's code and dependencies.
   1. `bot.properties`: This file contains configuration data, such as the bot's login credentials. A sampleChatBoxis located in the root of this project.
   1. `logging.properties` (optional): The configuration file for the Java Logging API.  A sample file is located in the root of this project.
1. Run ChatBox: `java -jar ChatBox-VERSION.jar`
1. Once the bot has fully started up, it will instruct you to press `Ctrl+Z`, then type the `bg` command. In Linux, this will move the program into the backgrouChatBox free up the shell.

# db.json

This is the file ChatBox usChatBoxpersist information, such as how many commands it has responded to and what rooms it has joined. It is located in the bot's workChatBoxrectory. The file will autoChatBoxlly be created if it doesn't exist.

# bot.properties

Contains various configuration settings for tChatBox. Open the sample "bot.propChatBox" file at the root of tChatBoxoject for a description of each setting.

ChatBox must be restarChatBox any of these settings are changed while ChatBox is running.

# Adding/Removing Commands

To add a command, create an instance of the [Command](https://github.com/JavaChat/ChatBox/blob/master/src/main/java/ChatBox/command/Command.java) interface and add it to the bot in the [main method](https://github.com/JavaChat/ChatBox/blob/master/src/main/java/ChatBox/Main.java).


# CLI Arguments

Argument | Description
-------- | -----------
--settings=PATH | The properties file that contains the bot's configuration settings, such as login credentials (defaults to "bot.properties").
--db=PATH | The path to a JSON file for storing all persistant data (defaults to "db.json").
--mock | Runs the bot using a mock chat connection for testing purposes.<br><br>A text file will be created in the root of the project for each chat room the bot is configured to connect to. These files are used to "send" messages to the mock chat rooms. To send a message, type your message into the text file and save it.<br><br>Messages are entered one per line. Multi-line messages can be entered by ending each line with a backslash until you reach the last line. You should only append onto the end of the file; do not delete anything. These files are re-created every time the program runs.<br><br>All messages that are sent to the mock chat room are displayed in stdout (this includes your messages and the bot's responses).
--quiet | If specified, tChatBox will not output a greeting message when it starts up.
--version | Prints the version of this program.
--help | Prints descriptions of each argument.

