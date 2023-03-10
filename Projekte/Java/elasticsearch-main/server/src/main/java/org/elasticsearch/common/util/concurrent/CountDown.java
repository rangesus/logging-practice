/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.common.util.concurrent;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A simple thread safe count-down class that in contrast to a {@link CountDownLatch}
 * never blocks. This class is useful if a certain action has to wait for N concurrent
 * tasks to return or a timeout to occur in order to proceed.
 */
public final class CountDown {

    private final AtomicInteger countDown;

    // Track originalCount because new CountDown(0) never reports completion which can be a leak if not handled specially
    // TODO fix the places that create an empty CountDown and then drop this, see #92196
    private final int originalCount;

    public CountDown(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("count must be greater or equal to 0 but was: " + count);
        }
        this.originalCount = count;
        this.countDown = new AtomicInteger(count);
    }

    /**
     * Decrements the count-down and returns <code>true</code> iff this call
     * reached zero otherwise <code>false</code>
     */
    public boolean countDown() {
        assert originalCount > 0;
        return countDown.getAndUpdate(current -> {
            assert current >= 0;
            return current == 0 ? 0 : current - 1;
        }) == 1;
    }

    /**
     * Fast forwards the count-down to zero and returns <code>true</code> iff
     * the count down reached zero with this fast forward call otherwise
     * <code>false</code>
     */
    public boolean fastForward() {
        assert originalCount > 0;
        assert countDown.get() >= 0;
        return countDown.getAndSet(0) > 0;
    }

    /**
     * Returns <code>true</code> iff the count-down has reached zero. Otherwise <code>false</code>
     */
    public boolean isCountedDown() {
        assert countDown.get() >= 0;
        return countDown.get() == 0;
    }
}
