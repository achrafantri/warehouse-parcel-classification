package com.example.achats.controller;

import com.example.achats.model.Devis;
import com.example.achats.service.DevisService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/devis")
public class DevisController {

    @Autowired
    private DevisService devisService;

    @GetMapping
    public List<Devis> getAllDevis() {
        return devisService.findAll();
    }

    @PostMapping
public Devis createDevis(@RequestBody Devis devis) {
        return devisService.save(devis);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Devis> getDevisById(@PathVariable Long id) {
        Devis devis = devisService.findById(id);
        return devis != null ? ResponseEntity.ok(devis) : ResponseEntity.notFound().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteDevis(@PathVariable Long id) {
        devisService.deleteById(id);
        return ResponseEntity.noContent().build();
    }
}
