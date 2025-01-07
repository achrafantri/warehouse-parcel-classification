package com.example.achats.controller;

import com.example.achats.model.Categorie;
import com.example.achats.service.CategorieService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
/**
 * Contrôleur REST API bech naamlou gestion des categorie.
 */
@RestController
@RequestMapping("/api/categorie")
public class CategorieController {
    /**
     *  3ayatna lel services
     */
    @Autowired
    private CategorieService categorieService;

    /**
     * ta3mel select lel les categories lkol
     */
    @GetMapping
    public List<Categorie> getAllCategories() {
        return categorieService.findAll();
    }

    /**
     * ta3mel select lel categore bel id
     */
    @GetMapping("/{id}")
    public ResponseEntity<Categorie> getCategorieById(@PathVariable Long id) {
        return categorieService.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /**
     * tasna3 categorie
     */
    @PostMapping
    public Categorie createCategorie(@RequestBody Categorie categorie) {
        return categorieService.save(categorie);
    }

    /**
     * tfasa5 categori
     */
    @PutMapping("/{id}")
    public ResponseEntity<Categorie> updateCategorie(@PathVariable Long id, @RequestBody Categorie categorie) {
        Categorie updatedCategorie = categorieService.update(id, categorie);
        return updatedCategorie != null ? ResponseEntity.ok(updatedCategorie) : ResponseEntity.notFound().build();
    }
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteCategorie(@PathVariable Long id) {
        categorieService.deleteById(id);
        return ResponseEntity.noContent().build();
    }
}
