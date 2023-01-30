package chatbox.chat;

/**
 * Thrown when the login credentials for connecting to a chat system are bad.
 * 
 */
@SuppressWarnings("serial")
public class InvalidCredentialsException extends RuntimeException {
	public InvalidCredentialsException() {
		super("Login credentials were rejected.");
	}
}
