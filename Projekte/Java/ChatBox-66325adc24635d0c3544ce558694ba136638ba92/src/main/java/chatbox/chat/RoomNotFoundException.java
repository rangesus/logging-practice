package chatbox.chat;

/**
 * Thrown when a request is sent to a room that doesn't exist.
 * 
 */
@SuppressWarnings("serial")
public class RoomNotFoundException extends RuntimeException {
	public RoomNotFoundException(int roomId) {
		super("Room " + roomId + " does not exist.");
	}
}
