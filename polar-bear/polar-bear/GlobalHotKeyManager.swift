//
//  GlobalHotKeyManager.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit
import Carbon

final class GlobalHotKeyManager {
    var onTrigger: (() -> Void)?

    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var carbonHotKeyRef: EventHotKeyRef?
    private var carbonEventHandler: EventHandlerRef?
    
    // Use Carbon hotkey for more reliable global detection
    private static var sharedInstance: GlobalHotKeyManager?

    func start() {
        stop()
        
        GlobalHotKeyManager.sharedInstance = self
        
        // Method 1: Carbon Hot Key (most reliable for global hotkeys)
        registerCarbonHotKey()
        
        // Method 2: NSEvent monitors as fallback
        // Global monitor - when app is NOT focused
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleKeyDown(event)
        }
        
        // Local monitor - when app IS focused
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if self?.isHotKey(event) == true {
                self?.triggerOnMainThread()
                return nil // Consume the event
            }
            return event
        }
    }

    func stop() {
        if let globalMonitor {
            NSEvent.removeMonitor(globalMonitor)
            self.globalMonitor = nil
        }
        if let localMonitor {
            NSEvent.removeMonitor(localMonitor)
            self.localMonitor = nil
        }
        unregisterCarbonHotKey()
        GlobalHotKeyManager.sharedInstance = nil
    }
    
    // MARK: - Carbon Hot Key Registration
    
    private func registerCarbonHotKey() {
        // Command + Option + C
        // Key code for 'C' is 8, Command is cmdKey, Option is optionKey
        let hotKeyID = EventHotKeyID(signature: OSType(0x504F4C52), id: 1) // "POLR"
        
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        
        let handler: EventHandlerUPP = { (_, event, _) -> OSStatus in
            GlobalHotKeyManager.sharedInstance?.triggerOnMainThread()
            return noErr
        }
        
        InstallEventHandler(GetApplicationEventTarget(), handler, 1, &eventType, nil, &carbonEventHandler)
        
        let modifiers: UInt32 = UInt32(cmdKey | optionKey)
        let keyCode: UInt32 = 8 // 'C' key
        
        RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &carbonHotKeyRef)
    }
    
    private func unregisterCarbonHotKey() {
        if let hotKeyRef = carbonHotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            carbonHotKeyRef = nil
        }
        if let handler = carbonEventHandler {
            RemoveEventHandler(handler)
            carbonEventHandler = nil
        }
    }
    
    // MARK: - Event Handling

    private func handleKeyDown(_ event: NSEvent) {
        if isHotKey(event) {
            triggerOnMainThread()
        }
    }
    
    private func isHotKey(_ event: NSEvent) -> Bool {
        if event.isARepeat { return false }
        let requiredFlags: NSEvent.ModifierFlags = [.command, .option]
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard flags.contains(requiredFlags) else { return false }
        guard event.charactersIgnoringModifiers?.lowercased() == "c" else { return false }
        return true
    }
    
    private func triggerOnMainThread() {
        if Thread.isMainThread {
            onTrigger?()
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.onTrigger?()
            }
        }
    }

    deinit {
        stop()
    }
}
