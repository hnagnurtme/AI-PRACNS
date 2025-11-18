package com.sagin.configuration;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import org.bson.codecs.configuration.CodecProvider;
import org.bson.codecs.configuration.CodecRegistry;
import org.bson.codecs.pojo.PojoCodecProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bson.codecs.configuration.CodecRegistries.fromProviders;
import static org.bson.codecs.configuration.CodecRegistries.fromRegistries;

/**
 * Static configuration for MongoDB, including constants, connection utilities,
 * and POJO Codec configuration.
 *
 * This class is final and non-instantiable.
 * It provides a thread-safe, singleton instance of MongoClientSettings.
 */
public final class MongoConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(MongoConfiguration.class);

    // --- Default / Constant Values ---
    public static final String DEFAULT_DATABASE_NAME = "network";
    public static final String NODES_COLLECTION = "network_nodes";
    public static final String USERS_COLLECTION = "network_users";

    // --- Singleton MongoClientSettings Instance ---
    /**
     * A pre-configured, singleton instance of MongoClientSettings.
     * This is created once when the class is loaded, ensuring high performance
     * and consistency. It includes the vital POJO Codec Registry.
     */
    private static final MongoClientSettings MONGO_CLIENT_SETTINGS = createMongoClientSettings();

    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private MongoConfiguration() {
    }

    // --- PUBLIC STATIC UTILITIES ---

    /**
     * Gets the MongoDB Connection String from environment variables.
     *
     * @return The connection string.
     * @throws IllegalStateException if the MONGODB_URI or MONGO_URI environment variable is not set.
     */
    public static String getConnectionString() {
        // Try MONGODB_URI first, then MONGO_URI
        String uri = System.getenv("MONGODB_URI");
        if (uri == null || uri.trim().isEmpty()) {
            uri = System.getenv("MONGO_URI");
        }

        if (uri == null || uri.trim().isEmpty()) {
            String errorMsg = "MongoDB URI not found in environment variables. " +
                            "Please set MONGODB_URI or MONGO_URI environment variable.";
            logger.error(errorMsg);
            throw new IllegalStateException(errorMsg);
        }

        logger.debug("Using MongoDB URI from environment variables");
        return uri;
    }

    /**
     * Gets the database name from environment variables, falling back to a
     * default.
     *
     * @return The database name.
     */
    public static String getDatabaseName() {
        return "network";
    }

    /**
     * Returns the pre-configured, singleton MongoClientSettings instance.
     *
     * @return The singleton MongoClientSettings.
     */
    public static MongoClientSettings getMongoClientSettings() {
        return MONGO_CLIENT_SETTINGS;
    }

    /**
     * Checks if the configuration is ready (i.e., if the connection string can be
     * loaded).
     *
     * @return true if ready, false otherwise.
     */
    public static boolean isConfigReady() {
        try {
            getConnectionString(); // This will throw if not set
            return true;
        } catch (IllegalStateException e) {
            return false;
        }
    }

    // --- INTERNAL INITIALIZATION METHODS ---

    /**
     * Internal method to create the MongoClientSettings when the class is
     * loaded.
     */
    private static MongoClientSettings createMongoClientSettings() {
        logger.debug("Creating MongoClientSettings singleton...");
        CodecRegistry pojoCodecRegistry = createPojoCodecRegistry();

        return MongoClientSettings.builder()
                .applyConnectionString(new ConnectionString(getConnectionString()))
                .codecRegistry(pojoCodecRegistry)
                .build();
    }

    /**
     * Creates a custom Codec Registry to allow the MongoDB Driver to
     * automatically map between BSON documents and Java objects (POJOs/Records).
     */
    private static CodecRegistry createPojoCodecRegistry() {
        // This provider enables automatic mapping for your Java classes.
        CodecProvider pojoCodecProvider = PojoCodecProvider.builder()
                .automatic(true) // Key setting!
                .build();

        // Combine the default codecs (for String, Int, etc.) with the new POJO
        // provider.
        return fromRegistries(
                MongoClientSettings.getDefaultCodecRegistry(),
                fromProviders(pojoCodecProvider));
    }
}