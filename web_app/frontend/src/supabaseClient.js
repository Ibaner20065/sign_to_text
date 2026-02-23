import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Custom storage adapter that handles both localStorage and sessionStorage
// based on a preference flag.
const customStorage = {
    getItem: (key) => {
        return localStorage.getItem(key) || sessionStorage.getItem(key);
    },
    setItem: (key, value) => {
        const rememberMe = localStorage.getItem('supabase_remember_me') === 'true';
        if (rememberMe) {
            localStorage.setItem(key, value);
            sessionStorage.removeItem(key);
        } else {
            sessionStorage.setItem(key, value);
            localStorage.removeItem(key);
        }
    },
    removeItem: (key) => {
        localStorage.removeItem(key);
        sessionStorage.removeItem(key);
    },
};

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
        storage: customStorage,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
    }
});

/**
 * Sets whether the session should be persistent (localStorage) or temporary (sessionStorage)
 * @param {boolean} value 
 */
export const setRememberMe = (value) => {
    localStorage.setItem('supabase_remember_me', value ? 'true' : 'false');
};
