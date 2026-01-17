//
//  AccessibilityService.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit
import ApplicationServices

final class AccessibilityService {
    func ensureProcessTrusted(prompt: Bool) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options = [key: prompt] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    func openAccessibilitySettings() {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") else {
            return
        }
        NSWorkspace.shared.open(url)
    }

    func focusedElement() -> AXUIElement? {
        let systemWide = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focused)
        guard error == .success, let element = focused as? AXUIElement else {
            return nil
        }
        return element
    }

    func readText(from element: AXUIElement) -> String? {
        if isSecureTextField(element) {
            NSSound.beep()
            return nil
        }

        var value: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &value)
        guard error == .success else {
            NSSound.beep()
            return nil
        }

        if let stringValue = value as? String {
            return stringValue
        }

        if let attributedValue = value as? NSAttributedString {
            return attributedValue.string
        }

        NSSound.beep()
        return nil
    }

    func writeText(_ text: String, to element: AXUIElement) -> Bool {
        let error = AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFTypeRef)
        return error == .success
    }

    func pasteTextViaKeyboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        let previousItems = pasteboard.pasteboardItems
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        sendKeyCombo(keyCode: 0, flags: .maskCommand) // Command + A
        sendKeyCombo(keyCode: 9, flags: .maskCommand) // Command + V

        if let previousItems {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                pasteboard.clearContents()
                pasteboard.writeObjects(previousItems)
            }
        }
    }

    private func isSecureTextField(_ element: AXUIElement) -> Bool {
        if let subrole = copyAttributeString(from: element, attribute: kAXSubroleAttribute as CFString),
           subrole == (kAXSecureTextFieldSubrole as String) {
            return true
        }

        return false
    }

    private func copyAttributeString(from element: AXUIElement, attribute: CFString) -> String? {
        var value: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(element, attribute, &value)
        guard error == .success else { return nil }
        return value as? String
    }

    private func sendKeyCombo(keyCode: CGKeyCode, flags: CGEventFlags) {
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return }
        let keyDown = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true)
        keyDown?.flags = flags
        let keyUp = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
        keyUp?.flags = flags
        keyDown?.post(tap: .cghidEventTap)
        keyUp?.post(tap: .cghidEventTap)
    }
}
