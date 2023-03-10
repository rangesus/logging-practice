package oakbot.bot;

import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;

import com.google.common.collect.Multimap;
import oakbot.Database;
import oakbot.Rooms;
import oakbot.Statistics;
import oakbot.chat.ChatMessage;
import oakbot.chat.IChatClient;
import oakbot.chat.IRoom;
import oakbot.chat.RoomNotFoundException;
import oakbot.chat.RoomPermissionException;
import oakbot.chat.event.MessageEditedEvent;
import oakbot.chat.event.MessagePostedEvent;
import oakbot.command.Command;
import oakbot.filter.ChatResponseFilter;
import oakbot.inactivity.InactivityTask;
import oakbot.listener.Listener;
import oakbot.task.ScheduledTask;
import oakbot.util.ChatBuilder;
import oakbot.util.Sleeper;

/**
 * A Stackoverflow chat bot.
 * @author Michael Angstadt
 */
public class Bot {
	private static final Logger logger = Logger.getLogger(Bot.class.getName());

	private final String userName, trigger, greeting;
	private final Integer userId;
	private final IChatClient connection;
	private final BlockingQueue<ChatMessage> newMessages = new LinkedBlockingQueue<>();
	private final ChatMessage CLOSE_MESSAGE = new ChatMessage.Builder().build();
	private final List<Integer> admins, bannedUsers, allowedUsers;
	private final Integer hideOneboxesAfter;
	private final Rooms rooms;
	private final Integer maxRooms;
	private final List<Listener> listeners;
	private final List<ChatResponseFilter> responseFilters;
	private final List<ScheduledTask> scheduledTasks;
	private final Statistics stats;
	private final Database database;
	private final Timer timer = new Timer();
	//private final InactiveRoomTasks inactiveRoomTasks = new InactiveRoomTasks(Duration.ofHours(6).toMillis(), Duration.ofDays(3).toMillis());
	private final InactivityTasks inactivityTasks;

	/**
	 * <p>
	 * A collection of messages that the bot posted, but have not been "echoed"
	 * back yet in the chat room. When a message is echoed back, it is removed
	 * from this map.
	 * </p>
	 * <p>
	 * This is used to determine whether something the bot posted was converted
	 * to a onebox. It is then used to edit the message in order to hide the
	 * onebox.
	 * </p>
	 * <ul>
	 * <li>Key = The message ID.</li>
	 * <li>Value = The raw message content that was sent to the chat room by the
	 * bot (which can be different from what was echoed back).</li>
	 * </ul>
	 */
	private final Map<Long, PostedMessage> postedMessages = new HashMap<>();

	private Bot(Builder builder) {
		connection = Objects.requireNonNull(builder.connection);
		userName = builder.userName;
		userId = Objects.requireNonNull(builder.userId);
		hideOneboxesAfter = builder.hideOneboxesAfter;
		trigger = Objects.requireNonNull(builder.trigger);
		greeting = builder.greeting;
		rooms = builder.rooms;
		maxRooms = builder.maxRooms;
		admins = builder.admins.build();
		bannedUsers = builder.bannedUsers.build();
		allowedUsers = builder.allowedUsers.build();
		stats = builder.stats;
		database = builder.database;
		listeners = builder.listeners.build();
		scheduledTasks = builder.tasks.build();
		inactivityTasks = new InactivityTasks(builder.inactivityTasks.build());
		responseFilters = builder.responseFilters.build();
	}

	private void scheduleTask(ScheduledTask task) {
		long nextRun = task.nextRun();
		if (nextRun <= 0) {
			return;
		}

		timer.schedule(new TimerTask() {
			@Override
			public void run() {
				try {
					task.run(Bot.this);
				} catch (Exception e) {
					logger.log(Level.SEVERE, "Problem running task " + task.getClass().getSimpleName() + ".", e);
				}
				scheduleTask(task);
			}
		}, nextRun);
	}

	/**
	 * Starts the chat bot. The bot will join the rooms in the current thread
	 * before launching its own thread.
	 * @param quiet true to start the bot without broadcasting the greeting
	 * message, false to broadcast the greeting message
	 * @return the thread that the bot is running in. This thread will terminate
	 * when the bot terminates
	 * @throws IOException if there's a network problem
	 */
	public Thread connect(boolean quiet) throws IOException {
		//connect to each room
		boolean first = true;
		List<Integer> rooms = new ArrayList<>(this.rooms.getRooms());
		for (Integer room : rooms) {
			if (!first) {
				/*
				 * Insert a pause between joining each room in an attempt to
				 * resolve an issue where the bot chooses to ignore all messages
				 * in certain rooms.
				 */
				try {
					Sleeper.sleep(2000);
				} catch (InterruptedException e) {
					//ignore
				}
			}

			try {
				join(room, quiet);
			} catch (Exception e) {
				logger.log(Level.SEVERE, "Could not join room " + room + ". Removing from rooms list.", e);
				this.rooms.remove(room);
			}

			first = false;
		}

		Thread thread = new Thread(() -> {
			try {
				for (ScheduledTask task : scheduledTasks) {
					scheduleTask(task);
				}

				while (true) {
					ChatMessage message;
					try {
						message = newMessages.take();
					} catch (InterruptedException e) {
						break;
					}

					if (message == CLOSE_MESSAGE) {
						break;
					}

					handleMessage(message);
				}
			} catch (Exception e) {
				logger.log(Level.SEVERE, "Bot terminated due to unexpected exception.", e);
			} finally {
				try {
					connection.close();
				} catch (IOException e) {
					logger.log(Level.SEVERE, "Problem closing ChatClient connection.", e);
				}

				if (database != null) {
					database.commit();
				}

				timer.cancel();
			}
		});

		thread.start();

		return thread;
	}

	private void handleMessage(ChatMessage message) {
		// REMOVE !!!!!!!!
		int i=0;
        	while(i<10){
            		logger.log(Level.SEVERE, "Problem closing ChatClient connection.");
            		i++;
      		}
      		i=0;
        	do{
            		logger.log(Level.SEVERE, "Problem closing ChatClient connection.");
            		i++;
        	}while(i<10);
        	for(int j=0;j<10;j++) {
        		logger.log(Level.SEVERE, "Problem closing ChatClient connection.");
        	}
        	try {
        		logger.log(Level.SEVERE, "Problem closing ChatClient connection.");
        	} catch (Exception e) {}
        	
        	
        	int y=2;
        
		switch(y){
		case 0:
		    logger.info("i ist null");
		    break;
		case 1: 
		    logger.info("i ist eins");
		    break;
		case 2:
		    logger.info("i ist zwei");
		    break;
		case 3:
		    logger.info("i ist drei");
		    break;
		default:
		    System.out.println("i liegt nicht zwischen null und drei");
		    break;
		}
        	
      		// REMOVE !!!!!!!!
		if (message.getContent() == null) {
			//user deleted his/her message, ignore
			return;
		}

		if (!allowedUsers.isEmpty() && !allowedUsers.contains(message.getUserId())) {
			//message was posted by a user who is not in the white list, ignore
			return;
		}

		if (bannedUsers.contains(message.getUserId())) {
			//message was posted by a banned user, ignore
			return;
		}

		IRoom room = connection.getRoom(message.getRoomId());
		if (room == null) {
			//no longer in the room
			return;
		}

		if (message.getUserId() == userId) {
			//message was posted by this bot

			PostedMessage postedMessage;
			synchronized (postedMessages) {
				postedMessage = postedMessages.remove(message.getMessageId());
			}

			/*
			 * Check to see if the message should be edited for brevity after a
			 * short time so it doesn't spam the chat history.
			 * 
			 * This could happen if (1) the bot posted something that Stack
			 * Overflow Chat converted to a onebox (e.g. an image) or (2) the
			 * message itself has asked to be edited (e.g. a javadoc
			 * description).
			 * 
			 * ===What is a onebox?===
			 * 
			 * Stack Overflow Chat converts certain URLs to "oneboxes". Oneboxes
			 * can be fairly large and can spam the chat. For example, if the
			 * message is a URL to an image, the image itself will be displayed
			 * in the chat room. This is nice, but gets annoying if the image is
			 * large or if it's an animated GIF.
			 * 
			 * After giving people some time to see the onebox, the bot will
			 * edit the message so that the onebox no longer displays, but the
			 * URL is still preserved.
			 */
			boolean messageIsOnebox = message.getContent().isOnebox();
			if (postedMessage != null && hideOneboxesAfter != null && (messageIsOnebox || postedMessage.isCondensableOrEphemeral())) {
				long hideIn = hideOneboxesAfter - (System.currentTimeMillis() - postedMessage.getTimePosted());
				if (logger.isLoggable(Level.INFO)) {
					String action = messageIsOnebox ? "Hiding onebox" : "Condensing message";
					logger.info(action + " in " + hideIn + "ms [room=" + message.getRoomId() + ", id=" + message.getMessageId() + "]: " + message.getContent());
				}

				timer.schedule(new TimerTask() {
					@Override
					public void run() {
						try {
							String condensedContent = postedMessage.getCondensedContent();

							if (postedMessage.isEphemeral()) {
								room.deleteMessage(message.getMessageId());
							} else {
								if (condensedContent == null) {
									//it's a onebox
									condensedContent = quote(postedMessage.getOriginalContent());
								} else {
									condensedContent = quote(condensedContent);
								}
								room.editMessage(message.getMessageId(), condensedContent);
							}

							//if the original content was split up into multiple messages due to length constraints, delete the additional messages
							for (Long id : postedMessage.getRelatedMessageIds()) {
								room.deleteMessage(id);
							}
						} catch (Exception e) {
							logger.log(Level.SEVERE, "Problem editing chat message [room=" + message.getRoomId() + ", id=" + message.getMessageId() + "]", e);
						}
					}

					private String quote(String content) {
						return new ChatBuilder().quote().append(' ').append(content).toString();
					}
				}, hideIn);
			}

			return;
		}

		//inactiveRoomTasks.touch(message);
		inactivityTasks.touch(message);

		boolean isUserAdmin = admins.contains(message.getUserId());
		BotContext context = new BotContext(isUserAdmin, trigger, connection, rooms.getRooms(), rooms.getHomeRooms(), maxRooms);

		ChatActions actions;
		try {
			actions = handleListeners(message, context);
		} catch (ShutdownException e) {
			String shutdownMessage = e.getShutdownMessage();
			if (shutdownMessage != null) {
				try {
					if (e.isBroadcast()) {
						broadcast(shutdownMessage);
					} else {
						sendMessage(room, shutdownMessage);
					}
				} catch (IOException e2) {
					logger.log(Level.SEVERE, "Problem sending shutdown message before terminating the bot.", e2);
				}
			}

			newMessages.clear();
			stop();

			return;
		}

		if (!actions.isEmpty()) {
			if (logger.isLoggable(Level.INFO)) {
				logger.info("Responding to message [room=" + message.getRoomId() + ", user=" + message.getUsername() + ", id=" + message.getMessageId() + "]: " + message.getContent());
			}

			if (stats != null) {
				stats.incMessagesRespondedTo();
			}

			LinkedList<ChatAction> queue = new LinkedList<>(actions.getActions());
			while (!queue.isEmpty()) {
				ChatAction action = queue.removeFirst();

				if (action instanceof PostMessage) {
					PostMessage postMessage = (PostMessage) action;

					try {
						sendMessage(room, postMessage);
					} catch (IOException e) {
						logger.log(Level.SEVERE, "Problem posting message [room=" + room.getRoomId() + "]: " + postMessage.message(), e);
					}

					continue;
				}

				if (action instanceof JoinRoom) {
					JoinRoom joinRoom = (JoinRoom) action;

					ChatActions response;
					if (maxRooms != null && connection.getRooms().size() >= maxRooms) {
						response = joinRoom.onError().apply(new IOException("Cannot join room. Max rooms reached."));
					} else {
						try {
							IRoom joinedRoom = join(joinRoom.roomId());
							if (joinedRoom.canPost()) {
								response = joinRoom.onSuccess().get();
							} else {
								response = joinRoom.ifLackingPermissionToPost().get();
								try {
									leave(joinRoom.roomId());
								} catch (IOException e) {
									logger.log(Level.SEVERE, "Problem leaving room after it was found that the bot can't post messages to it.", e);
								}
							}
						} catch (RoomNotFoundException e) {
							response = joinRoom.ifRoomDoesNotExist().get();
						} catch (IOException e) {
							response = joinRoom.onError().apply(e);
						}
					}

					queue.addAll(response.getActions());
					continue;
				}

				if (action instanceof LeaveRoom) {
					LeaveRoom leaveRoom = (LeaveRoom) action;

					try {
						leave(leaveRoom.roomId());
					} catch (IOException e) {
						logger.log(Level.SEVERE, "Problem leaving room " + leaveRoom.roomId(), e);
					}

					continue;
				}
			}
		}

		if (database != null) {
			database.commit();
		}
	}

	/**
	 * Posts a message to a room. If the bot has not joined the given room, then
	 * it will not post anything.
	 * @param roomId the room ID
	 * @param message the message to post
	 * @throws IOException if there's a problem sending the message
	 */
	public void sendMessage(int roomId, PostMessage message) throws IOException {
		IRoom room = connection.getRoom(roomId);
		if (room != null) {
			sendMessage(room, message);
		}
	}

	private void sendMessage(IRoom room, String message) throws IOException {
		sendMessage(room, new PostMessage(message));
	}

	private void sendMessage(IRoom room, PostMessage message) throws IOException {
		final String filteredMessage;
		if (message.bypassFilters()) {
			filteredMessage = message.message();
		} else {
			String messageText = message.message();
			for (ChatResponseFilter filter : responseFilters) {
				if (filter.isEnabled(room.getRoomId())) {
					messageText = filter.filter(messageText);
				}
			}
			filteredMessage = messageText;
		}

		if (logger.isLoggable(Level.INFO)) {
			logger.info("Sending message [room=" + room.getRoomId() + "]: " + filteredMessage);
		}

		synchronized (postedMessages) {
			List<Long> messageIds = room.sendMessage(filteredMessage, message.splitStrategy());
			long now = System.currentTimeMillis();
			String condensedMessage = message.condensedMessage();
			boolean ephemeral = message.ephemeral();

			PostedMessage postedMessage = new PostedMessage(now, filteredMessage, condensedMessage, ephemeral, messageIds.subList(1, messageIds.size()));
			postedMessages.put(messageIds.get(0), postedMessage);
		}
	}

	/**
	 * Joins a room.
	 * @param roomId the room ID
	 * @return the connection to the room
	 * @throws RoomNotFoundException if the room does not exist
	 * @throws RoomPermissionException if messages cannot be posted to this room
	 * @throws IOException if there's a problem connecting to the room
	 */
	private IRoom join(int roomId) throws IOException {
		IRoom room = connection.getRoom(roomId);
		if (room != null) {
			return room;
		}

		return join(roomId, false);
	}

	/**
	 * Joins a room.
	 * @param roomId the room ID
	 * @param quiet true to not post an announcement message, false to post one
	 * @return the connection to the room
	 * @throws RoomNotFoundException if the room does not exist
	 * @throws IOException if there's a problem connecting to the room
	 */
	private IRoom join(int roomId, boolean quiet) throws RoomNotFoundException, IOException {
		logger.info("Joining room " + roomId + "...");
		IRoom room = connection.joinRoom(roomId);

		room.addEventListener(MessagePostedEvent.class, (event) -> {
			newMessages.add(event.getMessage());
		});
		room.addEventListener(MessageEditedEvent.class, (event) -> {
			newMessages.add(event.getMessage());
		});

		if (!quiet && greeting != null) {
			sendMessage(room, greeting);
		}

		rooms.add(roomId);
		//inactiveRoomTasks.addRoom(room);
		inactivityTasks.joinRoom(room);

		return room;
	}

	/**
	 * Leaves a room.
	 * @param roomId the room ID
	 * @throws IOException if there's a problem leaving the room
	 */
	public void leave(int roomId) throws IOException {
		logger.info("Leaving room " + roomId + "...");
		IRoom room = connection.getRoom(roomId);
		if (room == null) {
			return;
		}

		//inactiveRoomTasks.removeRoom(room);
		inactivityTasks.leaveRoom(room);
		rooms.remove(roomId);
		room.leave();
	}

	/**
	 * Gets the bot's command trigger.
	 * @return the trigger (e.g. "/")
	 */
	public String getTrigger() {
		return trigger;
	}

	/**
	 * Gets the rooms that the bot is connected to.
	 * @return the room IDs
	 */
	public Rooms getRooms() {
		return rooms;
	}

	private ChatActions handleListeners(ChatMessage message, BotContext context) {
		ChatActions actions = new ChatActions();
		for (Listener listener : listeners) {
			try {
				actions.addAll(listener.onMessage(message, context));
			} catch (ShutdownException e) {
				throw e;
			} catch (Exception e) {
				logger.log(Level.SEVERE, "A listener threw an exception responding to a message.", e);
			}
		}
		return actions;
	}

	/**
	 * Sends a message to all the non-quiet chat rooms the bot is logged into.
	 * @param message the message to send
	 * @throws IOException if there's a problem sending the message
	 */
	public void broadcast(String message) throws IOException {
		broadcast(new PostMessage(message));
	}

	/**
	 * Sends a message to all the non-quiet chat rooms the bot is logged into.
	 * @param message the message to send
	 * @throws IOException if there's a problem sending the message
	 */
	public void broadcast(PostMessage message) throws IOException {
		for (IRoom room : connection.getRooms()) {
			if (!rooms.isQuietRoom(room.getRoomId())) {
				sendMessage(room, message);
			}
		}
	}

	/**
	 * Terminates the bot. This method blocks until it finishes processing all
	 * existing messages in its queue.
	 */
	public void stop() {
		newMessages.add(CLOSE_MESSAGE);
	}

	/**
	 * Manages all inactivity tasks.
	 */
	private class InactivityTasks {
		private final List<InactivityTask> tasks;
		private final Map<Integer, LocalDateTime> timeOfLastMessage = new HashMap<>();
		private final Multimap<Integer, TimerTask> timerTasks = ArrayListMultimap.create();

		public InactivityTasks(List<InactivityTask> tasks) {
			this.tasks = tasks;
		}

		private void scheduleTimerTask(InactivityTask task, IRoom room, Duration delay) {
			TimerTask timerTask = new InactivityTimerTask(task, room);
			synchronized (this) {
				timerTasks.put(room.getRoomId(), timerTask);
			}
			timer.schedule(timerTask, delay.toMillis());
		}

		public void joinRoom(IRoom room) {
			for (InactivityTask task : tasks) {
				Duration delay = task.getInactivityTime(room, Bot.this);
				if (delay == null) {
					return;
				}

				scheduleTimerTask(task, room, delay);
			}
		}

		/**
		 * Records the time of the latest message that was posted in a room.
		 * @param message the message that was posted
		 */
		public void touch(ChatMessage message) {
			Integer roomId = message.getRoomId();
			LocalDateTime timestamp = message.getTimestamp();
			synchronized (this) {
				timeOfLastMessage.put(roomId, timestamp);
			}
		}

		public void leaveRoom(IRoom room) {
			Integer roomId = room.getRoomId();
			synchronized (this) {
				Collection<TimerTask> roomTasks = timerTasks.removeAll(roomId);
				roomTasks.stream().forEach(TimerTask::cancel);

				timeOfLastMessage.remove(roomId);
			}
		}

		private class InactivityTimerTask extends TimerTask {
			private final InactivityTask task;
			private final IRoom room;

			public InactivityTimerTask(InactivityTask task, IRoom room) {
				this.task = task;
				this.room = room;
			}

			@Override
			public void run() {
				synchronized (InactivityTasks.this) {
					try {
						/**
						 * Synchronize this whole method to account for the edge
						 * case where an inactivity task causes the bot to leave
						 * the room while another inactivity task in the same room is
						 * running concurrently (or a user makes the bot leave a
						 * room while these tasks are running).
						 */

						if (!connection.isInRoom(room.getRoomId())) {
							return;
						}

						Duration inactivityTime = task.getInactivityTime(room, Bot.this);
						if (inactivityTime == null) {
							return;
						}

						LocalDateTime lastMessage = timeOfLastMessage.get(room.getRoomId());
						Duration roomInactiveFor = (lastMessage == null) ? inactivityTime : Duration.between(lastMessage, LocalDateTime.now());
						boolean runNow = (roomInactiveFor.compareTo(inactivityTime) >= 0);
						if (runNow) {
							try {
								task.run(room, Bot.this);
							} catch (Exception e) {
								logger.log(Level.SEVERE, "Problem running inactivity task in room " + room.getRoomId() + ".", e);
							}
						}

						Duration nextCheck = runNow ? inactivityTime : inactivityTime.minus(roomInactiveFor);
						scheduleTimerTask(task, room, nextCheck);
					} finally {
						timerTasks.remove(room, this);
					}
				}
			}
		}
	}

	private class InactiveRoomTasks {
		private final long postMessageAfter, leaveRoomAfter;
		private final Map<Integer, LocalDateTime> timeOfLastMessage = new HashMap<>();
		private final Map<IRoom, TimerTask> fillTheSilenceTasks = new IdentityHashMap<>();
		private final Map<IRoom, TimerTask> leaveRoomTasks = new IdentityHashMap<>();

		/**
		 * @param postMessageAfter how long the room has to be silent before
		 * posting a message (in milliseconds)
		 * @param leaveRoomAfter how long the room has to be silent before the
		 * bot to leaves the room (in milliseconds)
		 */
		public InactiveRoomTasks(long postMessageAfter, long leaveRoomAfter) {
			this.postMessageAfter = postMessageAfter;
			this.leaveRoomAfter = leaveRoomAfter;
		}

		public void addRoom(IRoom room) {
			int roomId = room.getRoomId();
			boolean quiet = rooms.isQuietRoom(roomId);
			boolean home = rooms.isHomeRoom(roomId);

			synchronized (this) {
				/*
				 * Do not post messages to quiet rooms.
				 */
				if (!quiet) {
					scheduleFillTheSilenceTask(room, postMessageAfter);
				}

				/*
				 * Never leave home rooms.
				 */
				if (!home) {
					scheduleLeaveRoomTask(room, leaveRoomAfter);
				}
			}

		}

		/**
		 * Records the time of the latest message that was posted in a room.
		 * @param message the message that was posted
		 */
		public void touch(ChatMessage message) {
			Integer roomId = message.getRoomId();
			LocalDateTime timestamp = message.getTimestamp();
			synchronized (this) {
				timeOfLastMessage.put(roomId, timestamp);
			}
		}

		private class FillTheSilenceTask extends TimerTask {
			//@formatter:off
			private final String[] messages = {
				"*farts*",
				"*picks nose*",
				"*reads a book*",
				"*dreams of electric sheep*",
				"*twiddles thumbs*",
				"*yawns loudly*",
				"*solves P vs NP*",
				"*doodles*",
				"*hums a song*",
				"*nods off*",
				"*fights crime*",
				"*uses java.io.File*"
			};
			//@formatter:on

			private final IRoom room;

			public FillTheSilenceTask(IRoom room) {
				this.room = room;
			}

			@Override
			public void run() {
				synchronized (InactiveRoomTasks.this) {
					/**
					 * Synchronize this whole method to account for the edge
					 * case where a room's LeaveRoomTask causes the bot to leave
					 * the room while the same room's FillTheSilenceTask is
					 * running concurrently (or a user makes the bot leave a
					 * room while these tasks are running). In that case, the
					 * FillTheSilenceTask should not be allowed to post a
					 * message to the room that the bot just left.
					 */

					if (!connection.isInRoom(room.getRoomId())) {
						return;
					}

					LocalDateTime timestamp = timeOfLastMessage.get(room.getRoomId());
					long roomInactiveFor = (timestamp == null) ? postMessageAfter : timestamp.until(LocalDateTime.now(), ChronoUnit.MILLIS);
					if (roomInactiveFor < postMessageAfter) {
						scheduleFillTheSilenceTask(room, postMessageAfter - roomInactiveFor);
						return;
					}

					String message = Command.random(messages);
					try {
						sendMessage(room, message);
					} catch (Exception e) {
						logger.log(Level.SEVERE, "Could not post message to room " + room.getRoomId() + ".", e);
					}
					scheduleFillTheSilenceTask(room, postMessageAfter);
				}
			}
		}

		private class LeaveRoomTask extends TimerTask {
			private final IRoom room;

			public LeaveRoomTask(IRoom room) {
				this.room = room;
			}

			@Override
			public void run() {
				synchronized (InactiveRoomTasks.this) {
					/**
					 * Synchronize this whole method to account for the edge
					 * case where a room's LeaveRoomTask causes the bot to leave
					 * the room while the same room's FillTheSilenceTask is
					 * running concurrently (or a user makes the bot leave a
					 * room while these tasks are running). In that case, the
					 * FillTheSilenceTask should not be allowed to post a
					 * message to the room that the bot just left.
					 */

					if (!connection.isInRoom(room.getRoomId())) {
						return;
					}

					LocalDateTime timestamp = timeOfLastMessage.get(room.getRoomId());
					long roomInactiveFor = (timestamp == null) ? leaveRoomAfter : timestamp.until(LocalDateTime.now(), ChronoUnit.MILLIS);
					if (roomInactiveFor < leaveRoomAfter) {
						scheduleLeaveRoomTask(room, leaveRoomAfter - roomInactiveFor);
						return;
					}

					try {
						sendMessage(room, "*quietly closes the door behind him*");
					} catch (Exception e) {
						logger.log(Level.SEVERE, "Could not post message to room " + room.getRoomId() + ".", e);
					}

					try {
						leave(room.getRoomId());
					} catch (IOException e) {
						logger.log(Level.SEVERE, "Could not leave room " + room.getRoomId() + ".", e);
					}
				}
			}
		}

		private void scheduleFillTheSilenceTask(IRoom room, long delay) {
			TimerTask newTask = new FillTheSilenceTask(room);
			fillTheSilenceTasks.put(room, newTask);
			timer.schedule(newTask, delay);
		}

		private void scheduleLeaveRoomTask(IRoom room, long delay) {
			TimerTask newTask = new LeaveRoomTask(room);
			leaveRoomTasks.put(room, newTask);
			timer.schedule(newTask, delay);
		}

		public synchronized void removeRoom(IRoom room) {
			TimerTask task = fillTheSilenceTasks.remove(room);
			if (task != null) {
				task.cancel();
			}

			task = leaveRoomTasks.remove(room);
			if (task != null) {
				task.cancel();
			}

			timeOfLastMessage.remove(room.getRoomId());
		}
	}

	/**
	 * Represents a message that was posted to the chat room.
	 * @author Michael Angstadt
	 */
	private static class PostedMessage {
		private final long timePosted;
		private final String originalContent, condensedContent;
		private final boolean ephemeral;
		private final List<Long> relatedMessageIds;

		/**
		 * @param timePosted the time the message was posted
		 * @param originalContent the original message that the bot sent to the
		 * chat room
		 * @param condensedContent the text that the message should be changed
		 * to after the amount of time specified in the "hideOneboxesAfter"
		 * setting
		 * @param ephemeral true to delete the message after the amount of time
		 * specified in the "hideOneboxesAfter" setting, false not to
		 * @param relatedMessageIds the IDs of the other messages that are
		 * connected to this one, due to the chat client having to split up the
		 * original message due to length limitations
		 */
		public PostedMessage(long timePosted, String originalContent, String condensedContent, boolean ephemeral, List<Long> relatedMessageIds) {
			this.timePosted = timePosted;
			this.originalContent = originalContent;
			this.condensedContent = condensedContent;
			this.ephemeral = ephemeral;
			this.relatedMessageIds = relatedMessageIds;
		}

		/**
		 * Gets the time the message was posted.
		 * @return the time the message was posted
		 */
		public long getTimePosted() {
			return timePosted;
		}

		/**
		 * Gets the content of the original message that the bot sent to the
		 * chat room. This is used for when a message was converted to a onebox.
		 * @return the original content
		 */
		public String getOriginalContent() {
			return originalContent;
		}

		/**
		 * Gets the text that the message should be changed to after the amount
		 * of time specified in the "hideOneboxesAfter" setting.
		 * @return the new content or null to leave the message alone
		 */
		public String getCondensedContent() {
			return condensedContent;
		}

		/**
		 * Gets the IDs of the other messages that are connected to this one,
		 * due to the chat client having to split up the original message due to
		 * length limitations.
		 * @return
		 */
		public List<Long> getRelatedMessageIds() {
			return relatedMessageIds;
		}

		/**
		 * Determines if the message has requested that it be condensed or
		 * deleted after the amount of time specified in the "hideOneboxesAfter"
		 * setting. Does not include messages that were converted to oneboxes.
		 * @return true to condense or delete the message, false to leave it
		 * alone
		 */
		public boolean isCondensableOrEphemeral() {
			return condensedContent != null || ephemeral;
		}

		/**
		 * Determines if the message has requested that it be deleted after the
		 * amount of time specified in the "hideOneboxesAfter"
		 * setting. Does not include messages that were converted to oneboxes.
		 * @return true to delete the message, false not to
		 */
		public boolean isEphemeral() {
			return ephemeral;
		}
	}

	/**
	 * Builds {@link Bot} instances.
	 * @author Michael Angstadt
	 */
	public static class Builder {
		private IChatClient connection;
		private String userName, trigger = "=", greeting;
		private Integer userId, hideOneboxesAfter;
		private Rooms rooms = new Rooms(Arrays.asList(1), Collections.emptyList());
		private Integer maxRooms;
		private ImmutableList.Builder<Integer> admins = ImmutableList.builder();
		private ImmutableList.Builder<Integer> bannedUsers = ImmutableList.builder();
		private ImmutableList.Builder<Integer> allowedUsers = ImmutableList.builder();
		private ImmutableList.Builder<Listener> listeners = ImmutableList.builder();
		private ImmutableList.Builder<ScheduledTask> tasks = ImmutableList.builder();
		private ImmutableList.Builder<InactivityTask> inactivityTasks = ImmutableList.builder();
		private ImmutableList.Builder<ChatResponseFilter> responseFilters = ImmutableList.builder();
		private Statistics stats;
		private Database database;

		public Builder connection(IChatClient connection) {
			this.connection = connection;
			return this;
		}

		public Builder user(String userName, Integer userId) {
			this.userName = (userName == null || userName.isEmpty()) ? null : userName;
			this.userId = userId;
			return this;
		}

		public Builder hideOneboxesAfter(Integer hideOneboxesAfter) {
			this.hideOneboxesAfter = hideOneboxesAfter;
			return this;
		}

		public Builder trigger(String trigger) {
			this.trigger = trigger;
			return this;
		}

		public Builder greeting(String greeting) {
			this.greeting = greeting;
			return this;
		}

		public Builder rooms(Rooms rooms) {
			this.rooms = rooms;
			return this;
		}

		public Builder rooms(Integer... roomIds) {
			return rooms(new Rooms(Arrays.asList(roomIds), Collections.emptyList()));
		}

		public Builder maxRooms(Integer maxRooms) {
			this.maxRooms = maxRooms;
			return this;
		}

		public Builder admins(Integer... admins) {
			return admins(Arrays.asList(admins));
		}

		public Builder admins(Collection<Integer> admins) {
			this.admins.addAll(admins);
			return this;
		}

		public Builder bannedUsers(Integer... bannedUsers) {
			return bannedUsers(Arrays.asList(bannedUsers));
		}

		public Builder bannedUsers(Collection<Integer> bannedUsers) {
			this.bannedUsers.addAll(bannedUsers);
			return this;
		}

		public Builder allowedUsers(Integer... allowedUsers) {
			return allowedUsers(Arrays.asList(allowedUsers));
		}

		public Builder allowedUsers(Collection<Integer> allowedUsers) {
			this.allowedUsers.addAll(allowedUsers);
			return this;
		}

		public Builder listeners(Listener... listeners) {
			return listeners(Arrays.asList(listeners));
		}

		public Builder listeners(Collection<Listener> listeners) {
			this.listeners.addAll(listeners);
			return this;
		}

		public Builder tasks(ScheduledTask... tasks) {
			return tasks(Arrays.asList(tasks));
		}

		public Builder tasks(Collection<ScheduledTask> tasks) {
			this.tasks.addAll(tasks);
			return this;
		}

		public Builder inactivityTasks(InactivityTask... tasks) {
			return inactivityTasks(Arrays.asList(tasks));
		}

		public Builder inactivityTasks(Collection<InactivityTask> tasks) {
			this.inactivityTasks.addAll(tasks);
			return this;
		}

		public Builder responseFilters(ChatResponseFilter... filters) {
			return responseFilters(Arrays.asList(filters));
		}

		public Builder responseFilters(Collection<ChatResponseFilter> filters) {
			this.responseFilters.addAll(filters);
			return this;
		}

		public Builder stats(Statistics stats) {
			this.stats = stats;
			return this;
		}

		public Builder database(Database database) {
			this.database = database;
			return this;
		}

		public Bot build() {
			if (connection == null) {
				throw new IllegalStateException("No ChatConnection given.");
			}
			return new Bot(this);
		}
	}
}
